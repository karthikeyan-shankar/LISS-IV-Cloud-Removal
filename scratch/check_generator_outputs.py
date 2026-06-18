import os
import sys
import numpy as np
import rasterio
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "models"))
from networks import Generator

def main():
    cloudy_path = "guwahati_cloudy_test.tif"
    model_weight_path = "models/generator_epoch_30.pth"
    clear_path = "data/raw/clear/guwahati_clear.tif"
    
    if not os.path.exists(cloudy_path):
        print(f"Error: {cloudy_path} not found")
        return
    if not os.path.exists(model_weight_path):
        print(f"Error: {model_weight_path} not found")
        return
        
    print("Reading files...")
    with rasterio.open(cloudy_path) as src:
        cloudy_data = src.read().astype(np.float32)
        bounds = src.bounds
        crs = src.crs
        
    with rasterio.open(clear_path) as ref_src:
        from rasterio.windows import from_bounds
        ref_window = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, ref_src.transform).round()
        ref_data = ref_src.read(window=ref_window, boundless=True, fill_value=0).astype(np.float32)
        
    if ref_data.shape[1] != cloudy_data.shape[1] or ref_data.shape[2] != cloudy_data.shape[2]:
        ref_data = ref_data[:, :cloudy_data.shape[1], :cloudy_data.shape[2]]
        
    print(f"Cloudy shape: {cloudy_data.shape}, min: {cloudy_data.min()}, max: {cloudy_data.max()}")
    print(f"Ref shape: {ref_data.shape}, min: {ref_data.min()}, max: {ref_data.max()}")
    
    scale = 1023.0
    print(f"Using constant scale: {scale}")
    
    c_norm = (cloudy_data / scale) * 2.0 - 1.0
    r_norm = (ref_data / scale) * 2.0 - 1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    gen = Generator(in_channels=6, out_channels=3)
    gen.load_state_dict(torch.load(model_weight_path, map_location=device))
    gen.to(device)
    gen.eval()
    
    input_tensor = torch.cat([
        torch.from_numpy(c_norm).unsqueeze(0),
        torch.from_numpy(r_norm).unsqueeze(0)
    ], dim=1).to(device)
    
    with torch.no_grad():
        output_tensor = gen(input_tensor).squeeze(0).cpu().numpy()
        
    reconstructed_data = ((output_tensor + 1.0) / 2.0) * scale
    print(f"Reconstructed min: {reconstructed_data.min()}, max: {reconstructed_data.max()}, mean: {reconstructed_data.mean()}")
    
    c_normalized = cloudy_data / (cloudy_data.max() if cloudy_data.max() > 0 else 1.0)
    cloud_mask = (c_normalized[0] > 0.6) & (c_normalized[1] > 0.6)
    
    print(f"Cloud mask positive pixels count: {np.sum(cloud_mask)}")
    
    masked_reconstructed = (cloud_mask * reconstructed_data) + ((1 - cloud_mask) * cloudy_data)
    print(f"Masked reconstructed mean: {masked_reconstructed.mean()}")
    
    # Save a small statistics summary
    with open("scratch/inference_stats.txt", "w") as f:
        f.write(f"Cloudy shape: {cloudy_data.shape}\n")
        f.write(f"Cloudy min/max: {cloudy_data.min()} / {cloudy_data.max()}\n")
        f.write(f"Ref min/max: {ref_data.min()} / {ref_data.max()}\n")
        f.write(f"Scale: {scale}\n")
        f.write(f"Reconstructed min/max/mean: {reconstructed_data.min()} / {reconstructed_data.max()} / {reconstructed_data.mean()}\n")
        f.write(f"Cloud mask positive: {np.sum(cloud_mask)}\n")
        f.write(f"Masked reconstructed mean: {masked_reconstructed.mean()}\n")
    print("Inference statistics written to scratch/inference_stats.txt")

if __name__ == "__main__":
    main()
