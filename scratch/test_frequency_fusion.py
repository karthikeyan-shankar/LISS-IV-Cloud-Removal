import os
import sys
import numpy as np
import rasterio
import torch
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "models"))
from networks import Generator
from test_tiled_inference import run_tiled_inference

def frequency_fusion(reconstructed, reference, ksize=15, sigma=3.0):
    """
    Blends the low frequencies of the reconstructed image (today's seasonal colors/illumination)
    with the high frequencies of the reference image (sharp geospatial structures, edges, and details).
    """
    bands, h, w = reconstructed.shape
    fused = np.zeros_like(reconstructed)
    
    for b in range(bands):
        recon_band = reconstructed[b]
        ref_band = reference[b]
        
        # Calculate low frequencies using a Gaussian filter
        recon_low = cv2.GaussianBlur(recon_band, (ksize, ksize), sigma)
        ref_low = cv2.GaussianBlur(ref_band, (ksize, ksize), sigma)
        
        # Calculate high frequencies of reference
        ref_high = ref_band - ref_low
        
        # Fuse: today's low frequency + historical high frequency
        fused_band = recon_low + ref_high
        fused[b] = fused_band
        
    return fused

def main():
    cloudy_path = "guwahati_cloudy_test.tif"
    model_weight_path = "models/generator_epoch_30.pth"
    clear_path = "data/raw/clear/guwahati_clear.tif"
    
    with rasterio.open(cloudy_path) as src:
        cloudy_data = src.read().astype(np.float32)
        bounds = src.bounds
        
    with rasterio.open(clear_path) as ref_src:
        from rasterio.windows import from_bounds
        ref_window = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, ref_src.transform).round()
        ref_data = ref_src.read(window=ref_window, boundless=True, fill_value=0).astype(np.float32)
        
    if ref_data.shape[1] != cloudy_data.shape[1] or ref_data.shape[2] != cloudy_data.shape[2]:
        ref_data = ref_data[:, :cloudy_data.shape[1], :cloudy_data.shape[2]]
        
    scale = 1023.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gen = Generator(in_channels=6, out_channels=3)
    gen.load_state_dict(torch.load(model_weight_path, map_location=device))
    gen.to(device)
    gen.eval()
    
    # Run tiled inference
    tiled_reconstructed = run_tiled_inference(gen, cloudy_data, ref_data, scale, device, patch_size=256, overlap=32)
    tiled_reconstructed = np.clip(tiled_reconstructed, 0, scale)
    
    # Run frequency fusion to restore sharp details in reconstructed regions
    fused_reconstructed = frequency_fusion(tiled_reconstructed, ref_data, ksize=15, sigma=3.0)
    fused_reconstructed = np.clip(fused_reconstructed, 0, scale)
    
    # Print stats
    print(f"Tiled Mean: {tiled_reconstructed.mean():.2f}")
    print(f"Fused Mean: {fused_reconstructed.mean():.2f}")
    
    # Save crops for checking
    h, w = cloudy_data.shape[1], cloudy_data.shape[2]
    cy, cx = h // 2, w // 2
    crop_size = 256
    
    tiled_crop = tiled_reconstructed[:, cy - crop_size//2 : cy + crop_size//2, cx - crop_size//2 : cx + crop_size//2]
    fused_crop = fused_reconstructed[:, cy - crop_size//2 : cy + crop_size//2, cx - crop_size//2 : cx + crop_size//2]
    ref_crop = ref_data[:, cy - crop_size//2 : cy + crop_size//2, cx - crop_size//2 : cx + crop_size//2]
    
    def save_crop_png(data, filename):
        def stretch_band(band):
            p99 = np.percentile(band, 99)
            p1 = np.percentile(band, 1)
            if p99 > p1:
                return np.clip((band - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)
            return np.clip((band / scale) * 255.0, 0, 255).astype(np.uint8)
        g = stretch_band(data[0])
        r = stretch_band(data[1])
        b = ((r.astype(np.float16) + g.astype(np.float16)) / 2).astype(np.uint8)
        bgr = cv2.merge([b, g, r])
        cv2.imwrite(filename, bgr)
        print(f"Saved {filename}")
        
    save_crop_png(tiled_crop, "scratch/tiled_crop_check.png")
    save_crop_png(fused_crop, "scratch/fused_crop_check.png")
    save_crop_png(ref_crop, "scratch/ref_crop_check.png")

if __name__ == "__main__":
    main()
