import os
import json
import random
import torch
import numpy as np
import cv2
import rasterio
from rasterio.features import shapes
from typing import Tuple, Optional, Any

from src.config import load_config, setup_logger
from src.utils import set_seed
from src.dataset import load_split
from src.model import build_model, get_device

logger = setup_logger(__name__)

def load_image(filepath: str) -> Tuple[np.ndarray, Any, Any]:
    """Loads a 2-channel SAR image (VV, VH) using rasterio."""
    with rasterio.open(filepath) as src:
        img = src.read([1, 2])
        transform = src.transform
        crs = src.crs
    
    img = img.astype(np.float32)
    for i in range(2):
        band_max = img[i].max()
        if band_max > 0:
            img[i] /= band_max
            
    return img, transform, crs

def clean_noise(mask: np.ndarray, threshold: int) -> np.ndarray:
    """Removes tiny clusters of pixels using OpenCV."""
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    clean_mask = np.zeros_like(mask_uint8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= threshold:
            clean_mask[labels == i] = 1
            
    return clean_mask

def generate_json(mask: np.ndarray, transform: Any, crs: Any, output_path: str) -> None:
    """Converts pixel masks to geospatial polygons and saves to JSON."""
    polygons = []
    
    for geom, val in shapes(mask, mask=(mask == 1), transform=transform):
        polygons.append({
            "type": "Feature",
            "properties": {
                "class": "oil_spill"
            },
            "geometry": geom
        })
        
    feature_collection = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": str(crs) if crs else "EPSG:4326"
            }
        },
        "features": polygons
    }
    
    with open(output_path, "w") as f:
        json.dump(feature_collection, f, indent=4)
        
    logger.info(f"Saved {len(polygons)} polygons to {output_path}")

def process_single_image(model: torch.nn.Module, device: torch.device, img_path: str, output_path: str, config: dict) -> None:
    """Run inference on a single SAR image, clean noise, and save oil spill polygons to JSON."""
    logger.info(f"Processing: {os.path.basename(img_path)}")
    try:
        img, transform, crs = load_image(img_path)
    except Exception as e:
        logger.error(f"Failed to load {img_path}: {e}")
        return
    
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    use_amp = config['training']['use_amp'] and device.type == 'cuda'
    
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(img_tensor)
            
    preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
    
    oil_class = config['model']['oil_class_index']
    oil_mask = (preds == oil_class).astype(np.uint8)
    initial_pixels = oil_mask.sum()
    
    if initial_pixels > 0:
        threshold = config['inference']['noise_threshold_px']
        clean_oil_mask = clean_noise(oil_mask, threshold)
        cleaned_pixels = clean_oil_mask.sum()
        
        if cleaned_pixels > 0:
            generate_json(clean_oil_mask, transform, crs, output_path)
        else:
            logger.info("All detected pixels were noise. No JSON generated.")
    else:
        logger.info("No oil spills detected.")

def predict(config_path: str = "config.yaml") -> None:
    """Batch processing mode for predictions."""
    config = load_config(config_path)
    set_seed(config['training']['seed'])
    
    output_dir = config['inference']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    device = get_device()
    config['model']['encoder_weights'] = None
    model = build_model(config, device)
    
    model_path = config['paths']['best_model']
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        logger.error(f"Cannot find model at {model_path}")
        return
        
    model.eval()

    try:
        test_images, _ = load_split(config['data']['manifest_path'], "test")
    except Exception as e:
        logger.error("Could not load split manifest.")
        return

    if not test_images:
        logger.error("Test split is empty.")
        return

    num_samples = min(config['inference']['batch_sample_count'], len(test_images))
    sample_images = random.sample(test_images, num_samples)
    logger.info(f"Selected {num_samples} images from test set.")
    
    for img_path in sample_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_json = os.path.join(output_dir, f"oil_spill_polygons_{base_name}.json")
        process_single_image(model, device, img_path, output_json, config)
        
    logger.info("Inference complete.")

if __name__ == "__main__":
    predict()
