import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from src.config import load_config, setup_logger
from src.utils import set_seed
from src.dataset import OilSpillDataset, load_split
from src.model import build_model, get_device

logger = setup_logger(__name__)

def train(config_path: str = "config.yaml") -> None:
    """Train the U-Net oil spill segmentation model with AMP and checkpoint resumption."""
    config = load_config(config_path)
    set_seed(config['training']['seed'])
    
    device = get_device()
    use_amp = config['training']['use_amp'] and device.type == "cuda"
    logger.info(f"AMP enabled: {use_amp}")

    manifest_path = config['data']['manifest_path']
    train_images, train_masks = load_split(manifest_path, "train")
    val_images, val_masks = load_split(manifest_path, "val")

    train_dataset = OilSpillDataset(train_images, train_masks, config, augment=True)
    val_dataset = OilSpillDataset(val_images, val_masks, config, augment=False)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(config, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    
    # Loss configuration
    loss_fn = smp.losses.DiceLoss(mode=config['training']['loss_mode'])
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    checkpoint_path = config['paths']['checkpoint']
    best_model_path = config['paths']['best_model']
    
    start_epoch = 0
    best_val_loss = float('inf')
    epochs = config['training']['epochs']

    # RESUME LOGIC
    if os.path.exists(checkpoint_path):
        logger.info(f"Found existing checkpoint. Loading {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed successfully! Starting from Epoch {start_epoch + 1} (Best Val Loss: {best_val_loss:.4f})")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                
                with torch.amp.autocast(device.type, enabled=use_amp):
                    outputs = model(imgs)
                    loss = loss_fn(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Update best validation loss and save best inference model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved new best model.")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss
        }, checkpoint_path)

if __name__ == "__main__":
    train()
