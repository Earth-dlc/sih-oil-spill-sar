import torch
from src.config import load_config, setup_logger
from src.utils import set_seed
from src.model import build_model, get_device

logger = setup_logger(__name__)

def smoke_test(config_path: str = "config.yaml") -> None:
    """Smoke-test: verify the U-Net model can perform a forward pass with dummy SAR input."""
    config = load_config(config_path)
    set_seed(config['training']['seed'])
    
    device = get_device()
    model = build_model(config, device)
    
    logger.info("Running smoke test...")
    
    # Create dummy tensor matching config dimensions
    c = config['model']['in_channels']
    h, w = config['preprocessing']['image_size']
    dummy_input = torch.randn(1, c, h, w).to(device)
    
    try:
        output = model(dummy_input)
        logger.info(f"Test forward pass successful. Output shape: {output.shape}")
        
        # Verify output shape is (1, num_classes, h, w)
        expected_shape = (1, config['model']['num_classes'], h, w)
        if output.shape == expected_shape:
            logger.info("Output shape is correct.")
        else:
            logger.warning(f"Output shape mismatch! Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")

if __name__ == "__main__":
    smoke_test()
