import torch
import segmentation_models_pytorch as smp
from typing import Dict, Any

from src.config import setup_logger

logger = setup_logger(__name__)

def build_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Initialize and return the U-Net model based on the configuration."""
    model_cfg = config['model']
    
    logger.info(f"Building {model_cfg['architecture']} model with {model_cfg['encoder']} encoder...")
    
    model = smp.Unet(
        encoder_name=model_cfg['encoder'],
        encoder_weights=model_cfg['encoder_weights'],
        in_channels=model_cfg['in_channels'],
        classes=model_cfg['num_classes'],
    )
    
    model.to(device)
    return model

def get_device() -> torch.device:
    """Return the available device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device.type.upper()}")
    return device
