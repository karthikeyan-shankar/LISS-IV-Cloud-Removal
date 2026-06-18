import os
import sys
import numpy as np
import rasterio
import torch
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "models"))
from networks import Generator
from test_tiled_inference import run_tiled_inference

def get_feathered_mask(cloudy_data, ref_data, scale=1023.0):
    # Normalize images to 0.0 - 1.0 for thresholding
    c_norm = cloudy_data / scale
    r_norm = ref_data / scale
    
    # 1. Cloud detection: bright in Green (band 0) and Red (band 1)
    # LISS-IV bands: 0=Green, 1=Red, 2=NIR
    cloud = (c_norm[0] > 0.35) & (c_norm[1] > 0.35)
    
    # 2. Shadow detection: where today's image is significantly darker than the history reference
    # and today's image is dark overall
    shadow_diff = (r_norm[1] - c_norm[1] > 0.08) & (c_norm[1] < 0.15)
    
    # Combine
    mask = (cloud | shadow_diff).astype(np.uint8)
    
    # Dilate mask to cover blurry cloud edges and shadow margins
    kernel = np.ones((7, 7), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=3)
    
    # Feather mask using Gaussian blur to ensure seamless transitions
    mask_feathered = cv2.GaussianBlur(mask_dilated.astype(np.float32), (21, 21), 7.0)
    
    # Ensure dimensions match
    mask_feathered = np.clip(mask_feathered, 0.0, 1.0)
    return mask_feathered

def frequency_fusion(reconstructed, reference, ksize=21, sigma=5.0):
    bands, h, w = reconstructed.shape
    fused = np.zeros_like(reconstructed)
    
    for b in range(bands):
        recon_band = reconstructed[b]
        ref_band = reference[b]
        
        # Low frequency component of reconstructed (today's translated color)
        recon_low = cv2.GaussianBlur(recon_band, (ksize, ksize), sigma)
        
        # Low frequency component of reference
        ref_low = cv2.GaussianBlur(ref_band, (ksize, ksize), sigma)
        
        # High frequency component of reference (sharp spatial features)
        ref_high = ref_band - ref_low
        
        # Fuse low frequency color with high frequency details
        fused[b] = recon_low + ref_high
        
    return fused

def main():
    cloudy_path = "guwahati_cloudy_test.tif"
    model_weight_path = "models/generator_epoch_30.pth"
    clear_path = "data/raw/clear/guwahati_clear.tif"
    
    with rasterio.open(cloudy_path) as src:
        cloudy_data = src.read().astype(np.float32)
        bounds = src.bounds
        profile = src.profile
        
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
    
    # 1. Run tiled inference to get color translation in cloudy areas
    tiled_reconstructed = run_tiled_inference(gen, cloudy_data, ref_data, scale, device, patch_size=256, overlap=32)
    tiled_reconstructed = np.clip(tiled_reconstructed, 0, scale)
    
    # 2. Run frequency fusion to inject reference details into model low-frequencies
    fused_reconstructed = frequency_fusion(tiled_reconstructed, ref_data, ksize=21, sigma=5.0)
    fused_reconstructed = np.clip(fused_reconstructed, 0, scale)
    
    # 3. Generate feathered mask
    mask = get_feathered_mask(cloudy_data, ref_data, scale=scale)
    
    # 4. Blend: keep original clear pixels, use fused pixels in cloudy/shadowed areas
    final_output = np.zeros_like(cloudy_data)
    for b in range(3):
        final_output[b] = (1.0 - mask) * cloudy_data[b] + mask * fused_reconstructed[b]
        
    final_output = np.clip(final_output, 0, scale).astype(profile['dtype'])
    
    # Save the output file
    out_path = "scratch/guwahati_hybrid_fused.tif"
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(final_output)
        
    print(f"Hybrid output saved to {out_path}")
    
    # Save a crop to check
    h_dim, w_dim = cloudy_data.shape[1], cloudy_data.shape[2]
    cy, cx = h_dim // 2, w_dim // 2
    crop_size = 256
    
    cloudy_crop = cloudy_data[:, cy - crop_size//2 : cy + crop_size//2, cx - crop_size//2 : cx + crop_size//2]
    final_crop = final_output[:, cy - crop_size//2 : cy + crop_size//2, cx - crop_size//2 : cx + crop_size//2]
    
    # Helper to save as PNG
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
        
    save_crop_png(cloudy_crop, "scratch/cloudy_crop_check.png")
    save_crop_png(final_crop, "scratch/final_crop_check.png")

if __name__ == "__main__":
    main()
