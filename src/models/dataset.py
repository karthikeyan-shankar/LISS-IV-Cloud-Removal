import os
import glob
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset

class LISS4Dataset(Dataset):
    """
    Self-Supervised PyTorch Dataset for LISS-IV Cloud Reconstruction.
    Since we don't have "ground truth" clear images for cloudy days, 
    we train by taking 100% CLEAR patches, dynamically simulating clouds on them,
    and training the GAN to reconstruct the original clear patch using 
    a historical reference.
    """
    def __init__(self, clear_dir, transform=None):
        self.clear_dir = clear_dir
        self.transform = transform
        
        # We train on the 3500+ clear patches
        self.clear_patches = sorted(glob.glob(os.path.join(clear_dir, "*.tif")))
        
    def __len__(self):
        return len(self.clear_patches)
        
    def simulate_clouds(self, img):
        """
        Dynamically draws random white cloudy blobs and shadows on the clear image.
        img shape: (3, H, W) normalized to 0.0 - 1.0.
        """
        cloudy_img = img.copy()
        h, w = img.shape[1], img.shape[2]
        
        # Create a black mask of same size
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Draw 1 to 3 random cloud blobs
        num_clouds = np.random.randint(1, 4)
        for _ in range(num_clouds):
            # Random center coordinates
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            # Random cloud radius
            r = np.random.randint(20, 60)
            
            # Create a coordinate grid
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Create smooth feathering boundary for the cloud using a sigmoid shape
            blob_mask = np.clip(1.0 - (dist / r), 0, 1)
            blob_mask = np.smoothstep(0.0, 1.0, blob_mask)
            
            mask = np.maximum(mask, blob_mask)
            
        # Add the clouds (bright white pixels)
        # We blend white (1.0) into the image using the mask
        for b in range(3): # bands
            cloudy_img[b] = (mask * 1.0) + ((1.0 - mask) * cloudy_img[b])
            
        # Add corresponding cloud shadows (slightly shifted to the bottom-right)
        # Shift mask by 15-30 pixels to create shadow location
        shadow_mask = np.roll(mask, shift=(np.random.randint(10, 25), np.random.randint(10, 25)), axis=(0, 1))
        # Dim the pixels under the shadow mask by 30%
        for b in range(3):
            cloudy_img[b] = np.where(shadow_mask > 0.2, cloudy_img[b] * (1.0 - 0.3 * shadow_mask), cloudy_img[b])
            
        return cloudy_img, mask

    def __getitem__(self, idx):
        clear_path = self.clear_patches[idx]
        
        # Read the Clear GeoTIFF
        with rasterio.open(clear_path) as src:
            target_clear = src.read().astype(np.float32) # (3, H, W)
            
        # Scale to 0.0 - 1.0 (LISS-IV is 10-bit data with values up to 1023)
        scale = 1023.0
        target_clear /= scale
        
        # Dynamically simulate clouds on the target image
        cloudy_input, cloud_mask = self.simulate_clouds(target_clear)
        
        # Create a mock historical reference image
        # In remote sensing, seasons change, so we add a slight color/noise shift 
        # to the target clear image to act as the "historical reference"
        history_ref = target_clear + np.random.normal(0, 0.03, target_clear.shape).astype(np.float32)
        history_ref = np.clip(history_ref, 0.0, 1.0)
        
        # Apply data augmentations if any
        if self.transform:
            # Albumentations expects HWC
            target_hwc = np.transpose(target_clear, (1, 2, 0))
            cloudy_hwc = np.transpose(cloudy_input, (1, 2, 0))
            history_hwc = np.transpose(history_ref, (1, 2, 0))
            
            augmented = self.transform(image=target_hwc, mask=cloudy_hwc, ref=history_hwc)
            target_clear = np.transpose(augmented['image'], (2, 0, 1))
            cloudy_input = np.transpose(augmented['mask'], (2, 0, 1))
            history_ref = np.transpose(augmented['ref'], (2, 0, 1))
            
        return {
            'cloudy': torch.from_numpy(cloudy_input),      # (3, H, W) - Input 1
            'history': torch.from_numpy(history_ref),      # (3, H, W) - Input 2
            'clear': torch.from_numpy(target_clear),        # (3, H, W) - Target Ground Truth
            'mask': torch.from_numpy(cloud_mask)           # (1, H, W) - Cloud Mask
        }

# Custom smoothstep function for numpy
if not hasattr(np, 'smoothstep'):
    def smoothstep(edge0, edge1, x):
        x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    np.smoothstep = smoothstep

if __name__ == "__main__":
    print("Self-supervised dataset loader initialized!")
