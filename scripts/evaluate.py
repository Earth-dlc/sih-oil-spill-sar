import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.amp import autocast

from src.config import load_config, setup_logger
from src.utils import set_seed
from src.dataset import OilSpillDataset, load_split
from src.model import build_model, get_device

logger = setup_logger(__name__)

def evaluate(config_path: str = "config.yaml") -> None:
    """Evaluate the trained model on the held-out test set, reporting IoU, Dice, and loss."""
    config = load_config(config_path)
    set_seed(config['training']['seed'])
    
    device = get_device()
    use_amp = config['training']['use_amp'] and device.type == "cuda"
    logger.info(f"Evaluating (AMP enabled: {use_amp})")

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------
    manifest_path = config['data']['manifest_path']
    test_images, test_masks = load_split(manifest_path, "test")
    logger.info(f"Test set size: {len(test_images)} images")

    test_dataset = OilSpillDataset(test_images, test_masks, config, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    # Temporarily override encoder weights for initialization before loading checkpoint
    config['model']['encoder_weights'] = None
    model = build_model(config, device)
    
    model_path = config['paths']['best_model']
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded weights from {model_path}")
    except FileNotFoundError:
        logger.error(f"Could not find model weights at {model_path}. Train the model first.")
        return
        
    model.to(device)
    model.eval()

    loss_fn = smp.losses.DiceLoss(mode=config['training']['loss_mode'])

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------
    total_loss = 0.0
    total_images = 0
    oil_gt_count = 0
    no_oil_gt_count = 0
    image_wise_iou_sum = 0.0
    global_inter = 0.0
    global_union = 0.0
    oil_dice_sum = 0.0
    oil_dice_count = 0
    
    num_classes = config['model']['num_classes']
    oil_class = config['model']['oil_class_index']

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}...")

            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(device.type, enabled=use_amp):
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            for i in range(masks.size(0)):
                total_images += 1
                pred = preds[i]
                gt = masks[i]

                has_oil = (gt == oil_class).any().item()
                if has_oil:
                    oil_gt_count += 1
                else:
                    no_oil_gt_count += 1

                per_class_iou = []
                for cls in range(num_classes):
                    p_c = (pred == cls)
                    g_c = (gt == cls)
                    inter = (p_c & g_c).sum().item()
                    union = (p_c | g_c).sum().item()
                    if union > 0:
                        per_class_iou.append(inter / union)
                if per_class_iou:
                    image_wise_iou_sum += sum(per_class_iou) / len(per_class_iou)

                pred_oil = (pred == oil_class)
                gt_oil = (gt == oil_class)
                g_inter = (pred_oil & gt_oil).sum().item()
                g_union = (pred_oil | gt_oil).sum().item()
                global_inter += g_inter
                global_union += g_union

                if has_oil:
                    oil_sum = pred_oil.sum().item() + gt_oil.sum().item()
                    if oil_sum > 0:
                        dice = (2.0 * g_inter) / oil_sum
                        oil_dice_sum += dice
                        oil_dice_count += 1

    # -------------------------------------------------------------------------
    # FINAL METRICS
    # -------------------------------------------------------------------------
    avg_loss = total_loss / len(test_loader)
    mean_img_iou = image_wise_iou_sum / total_images if total_images > 0 else 0.0
    global_micro_iou = global_inter / global_union if global_union > 0 else 0.0
    oil_mean_dice = oil_dice_sum / oil_dice_count if oil_dice_count > 0 else 0.0

    print("\n" + "=" * 55)
    print("FINAL HOLDOUT TEST EVALUATION")
    print("=" * 55)
    print(f"Total Test Images          : {total_images}")
    print(f"GT Oil Images              : {oil_gt_count}")
    print(f"GT No-Oil Images           : {no_oil_gt_count}")
    print("-" * 55)
    print(f"Average Dice Loss          : {avg_loss:.4f}")
    print(f"Mean Image-wise IoU        : {mean_img_iou:.4f}  ({mean_img_iou * 100:.2f}%)")
    print(f"Global/Micro IoU (Oil)     : {global_micro_iou:.4f}  ({global_micro_iou * 100:.2f}%)")
    print(f"Oil Spill Mean Dice        : {oil_mean_dice:.4f}  ({oil_mean_dice * 100:.2f}%)")
    print("=" * 55)

if __name__ == "__main__":
    evaluate()
