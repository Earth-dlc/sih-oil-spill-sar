import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import tifffile
from PIL import Image
from typing import Tuple, List

from src.config import setup_logger

logger = setup_logger(__name__)

class OilSpillDataset(Dataset):
    """PyTorch Dataset for loading 2-channel SAR images and segmentation masks.

    Handles VV+VH dual-polarization GeoTIFF loading, per-image max-scaling,
    resizing to a fixed resolution, and optional geometric augmentation.
    """
    def __init__(self, image_paths: List[str], mask_paths: List[str], config: dict, augment: bool = False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.config = config
        
        self.img_size = self.config['preprocessing']['image_size']

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load, normalize, resize, and optionally augment a single image-mask pair."""
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load SAR image (VV + VH dual-polarization)
        image = tifffile.imread(img_path).astype(np.float32)

        # Ensure channel-first format (C, H, W)
        if image.shape[-1] == 2:
            image = np.transpose(image, (2, 0, 1))   # (H, W, 2) → (2, H, W)

        # Per-image max-scaling: deliberate choice for SAR speckle noise
        img_max = image.max()
        if img_max > 0:
            image = image / img_max

        # Load mask
        mask = np.array(Image.open(mask_path)).astype(np.longlong)

        # Convert to tensors
        img_tensor = torch.tensor(image)
        mask_tensor = torch.tensor(mask)

        # Resize for DataLoader batch collation
        img_tensor = TF.resize(img_tensor, self.img_size,
                                 interpolation=InterpolationMode.BILINEAR,
                                 antialias=True)
        mask_tensor = TF.resize(mask_tensor.unsqueeze(0), self.img_size,
                                 interpolation=InterpolationMode.NEAREST).squeeze(0)

        # Augmentation
        if self.augment and self.config['augmentation']['enabled']:
            if random.random() < self.config['augmentation']['horizontal_flip_prob']:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)
            if random.random() < self.config['augmentation']['vertical_flip_prob']:
                img_tensor = TF.vflip(img_tensor)
                mask_tensor = TF.vflip(mask_tensor)

        return img_tensor, mask_tensor


def load_split(manifest_path: str, split_name: str) -> Tuple[List[str], List[str]]:
    """Load image and mask file paths for a given split from the JSON manifest.

    Args:
        manifest_path: Path to the split_manifest.json file.
        split_name: One of 'train', 'val', or 'test'.

    Returns:
        Tuple of (image_paths, mask_paths) as lists of strings.
    """
    logger.info(f"Loading split '{split_name}' from {manifest_path}...")
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        # Normalize paths for cross-platform compatibility
        images = [os.path.normpath(p) for p in manifest[split_name]['images']]
        masks = [os.path.normpath(p) for p in manifest[split_name]['masks']]
        return images, masks
    except Exception as e:
        logger.error(f"Failed to load manifest split: {e}")
        raise
