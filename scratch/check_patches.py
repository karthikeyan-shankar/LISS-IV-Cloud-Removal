import os
import glob
import rasterio
import numpy as np

def main():
    patches_dir = "data/patches/clear"
    if not os.path.exists(patches_dir):
        print(f"Error: {patches_dir} not found")
        return
        
    patch_files = glob.glob(os.path.join(patches_dir, "*.tif"))
    if not patch_files:
        print(f"No patches found in {patches_dir}")
        return
        
    print(f"Checking {len(patch_files)} patches...")
    max_vals = []
    min_vals = []
    mean_vals = []
    
    # Check first 50 patches
    for f in patch_files[:50]:
        with rasterio.open(f) as src:
            data = src.read()
            max_vals.append(data.max())
            min_vals.append(data.min())
            mean_vals.append(data.mean())
            
    print(f"Stats over first 50 patches:")
    print(f"Min of min: {min(min_vals)}, Max of min: {max(min_vals)}")
    print(f"Min of max: {min(max_vals)}, Max of max: {max(max_vals)}")
    print(f"Min of mean: {min(mean_vals)}, Max of mean: {max(mean_vals)}")
    
    # Count how many patches have max value > 255
    above_255_count = 0
    all_max_vals = []
    for f in patch_files:
        with rasterio.open(f) as src:
            mx = src.read().max()
            all_max_vals.append(mx)
            if mx > 255:
                above_255_count += 1
                
    print(f"Total patches: {len(patch_files)}")
    print(f"Patches with max > 255: {above_255_count} ({above_255_count/len(patch_files)*100:.2f}%)")
    print(f"Overall absolute min/max across all patches: {min(all_max_vals)} / {max(all_max_vals)}")

if __name__ == "__main__":
    main()
