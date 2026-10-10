import os
import random
import numpy as np
import torch
from src.config import setup_logger

logger = setup_logger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across numpy, random, and PyTorch."""
    logger.info(f"Setting random seed to {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
