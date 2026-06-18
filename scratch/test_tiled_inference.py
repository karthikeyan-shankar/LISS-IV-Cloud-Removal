import os
import sys
import numpy as np
import rasterio
import torch
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "models"))
from networks import Generator

def run_tiled_inference(gen, cloudy_data, ref_data, scale, device, patch_size=256, overlap=32):
    _, h, w = cloudy_data.shape
    output = np.zeros_like(cloudy_data)
    weight_padded = np.zeros((h, w), dtype=np.float32)
    
    # Create 2D linear blending weight window
    window_1d = np.ones(patch_size, dtype=np.float32)
    if overlap > 0:
        feather = np.linspace(0, 1, overlap)
        window_1d[:overlap] = feather
        window_1d[-overlap:] = feather[::-1]
    window_2d = np.outer(window_1d, window_1d)
    
    stride = patch_size - overlap
    
    # Pad inputs if not divisible
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride
    
    c_padded = np.pad(cloudy_data, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
    r_padded = np.pad(ref_data, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
    
    h_padded, w_padded = c_padded.shape[1], c_padded.shape[2]
    out_padded = np.zeros_like(c_padded)
    weight_padded = np.zeros((h_padded, w_padded), dtype=np.float32)
    
    for y in range(0, h_padded - patch_size + 1, stride):
        for x in range(0, w_padded - patch_size + 1, stride):
            c_crop = c_padded[:, y:y+patch_size, x:x+patch_size]
            r_crop = r_padded[:, y:y+patch_size, x:x+patch_size]
            
            c_norm = (c_crop / scale) * 2.0 - 1.0
            r_norm = (r_crop / scale) * 2.0 - 1.0
            
            input_tensor = torch.cat([
                torch.from_numpy(c_norm).unsqueeze(0),
                torch.from_numpy(r_norm).unsqueeze(0)
            ], dim=1).to(device)
            
            with torch.no_grad():
                output_tensor = gen(input_tensor).squeeze(0).cpu().numpy()
                
            crop_out = ((output_tensor + 1.0) / 2.0) * scale
            
            for b in range(3):
                out_padded[b, y:y+patch_size, x:x+patch_size] += crop_out[b] * window_2d
            weight_padded[y:y+patch_size, x:x+patch_size] += window_2d
            
    weight_padded = np.maximum(weight_padded, 1e-8)
    for b in range(3):
        out_padded[b] /= weight_padded
        
    return out_padded[:, :h, :w]

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
    
    # 1. Run direct inference
    c_norm = (cloudy_data / scale) * 2.0 - 1.0
    r_norm = (ref_data / scale) * 2.0 - 1.0
    input_tensor = torch.cat([
        torch.from_numpy(c_norm).unsqueeze(0),
        torch.from_numpy(r_norm).unsqueeze(0)
    ], dim=1).to(device)
    with torch.no_grad():
        direct_out = gen(input_tensor).squeeze(0).cpu().numpy()
    direct_reconstructed = ((direct_out + 1.0) / 2.0) * scale
    direct_reconstructed = np.clip(direct_reconstructed, 0, scale)
    
    # 2. Run tiled inference
    tiled_reconstructed = run_tiled_inference(gen, cloudy_data, ref_data, scale, device, patch_size=256, overlap=32)
    tiled_reconstructed = np.clip(tiled_reconstructed, 0, scale)
    
    # Print comparison stats
    print(f"Direct - Min: {direct_reconstructed.min():.2f}, Max: {direct_reconstructed.max():.2f}, Mean: {direct_reconstructed.mean():.2f}")
    print(f"Tiled  - Min: {tiled_reconstructed.min():.2f}, Max: {tiled_reconstructed.max():.2f}, Mean: {tiled_reconstructed.mean():.2f}")
    
    # Save a patch crop of both to compare texture quality
    # Crop a 256x256 region from the center
    h, w = cloudy_data.shape[1], cloudy_data.shape[2]
    cy, cx = h // 2, w // 2
    crop_size = 256
    
    direct_crop = direct_reconstructed[:, cy - crop_size//2 : cy + crop_size//2, cx - crop_size//2 : cx + crop_size//2]
    tiled_crop = tiled_reconstructed[:, cy - crop_size//2 : cy + crop_size//2, cx - crop_size//2 : cx + crop_size//2]
    
    # Save crops as PNG to check checkerboard artifact reduction
    def save_crop_png(data, filename):
        # Apply 1%-99% contrast stretching
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
        
    save_crop_png(direct_crop, "scratch/direct_crop_check.png")
    save_crop_png(tiled_crop, "scratch/tiled_crop_check.png")

if __name__ == "__main__":
    main()
