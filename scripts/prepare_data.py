import os
import json
import glob
import random
from pathlib import PurePosixPath

from src.config import load_config, setup_logger
from src.utils import set_seed

logger = setup_logger(__name__)

def create_manifest(config_path: str = "config.yaml") -> None:
    """Generate a seeded train/val/test split manifest from the dataset directory."""
    config = load_config(config_path)
    set_seed(config['training']['seed'])
    
    data_cfg = config['data']
    data_dir = data_cfg['dataset_dir']
    
    # Use PurePosixPath to ensure cross-platform compatibility for saved paths
    images_pattern = os.path.join(data_dir, data_cfg['images_subdir'], "*.tif")
    masks_pattern = os.path.join(data_dir, data_cfg['masks_subdir'], "*.tif")
    
    images = sorted(glob.glob(images_pattern))
    masks = sorted(glob.glob(masks_pattern))
    
    if not images or not masks:
        logger.error(f"No data found in {data_dir}. Did you download it?")
        return

    data = list(zip(images, masks))
    random.shuffle(data)

    n = len(data)
    train_end = int(n * data_cfg['split_ratios']['train'])
    val_end = train_end + int(n * data_cfg['split_ratios']['val'])

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Convert paths to forward slashes for cross-platform JSON
    def to_posix(path_str: str) -> str:
        return str(PurePosixPath(path_str).as_posix())

    manifest = {
        "train": {"images": [to_posix(x[0]) for x in train_data], "masks": [to_posix(x[1]) for x in train_data]},
        "val": {"images": [to_posix(x[0]) for x in val_data], "masks": [to_posix(x[1]) for x in val_data]},
        "test": {"images": [to_posix(x[0]) for x in test_data], "masks": [to_posix(x[1]) for x in test_data]}
    }

    output_file = data_cfg['manifest_path']
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    logger.info(f"Manifest created at {output_file} with split:")
    logger.info(f"  Train: {len(train_data)} images")
    logger.info(f"  Val: {len(val_data)} images")
    logger.info(f"  Test: {len(test_data)} images")

if __name__ == "__main__":
    create_manifest()
