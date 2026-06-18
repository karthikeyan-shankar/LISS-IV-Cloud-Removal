import os
import numpy as np
import rasterio
import cv2

def main():
    cloudy_path = "guwahati_cloudy_test.tif"
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
        
    c_max = np.percentile(cloudy_data, 99)
    r_max = np.percentile(ref_data, 99)
    if c_max <= 0: c_max = 1.0
    if r_max <= 0: r_max = 1.0
    c_norm = cloudy_data / c_max
    r_norm = ref_data / r_max
    
    # Cloud threshold (lower to catch thin cloud margins)
    cloud = (c_norm[0] > 0.45) & (c_norm[1] > 0.45)
    
    # Shadow threshold: today is darker than history clear in Green
    shadow = (r_norm[0] - c_norm[0] > 0.08)
    
    mask = (cloud | shadow).astype(np.uint8)
    
    # Dilate mask
    kernel = np.ones((11, 11), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=3)
    
    print(f"Cloud pixels: {np.sum(cloud)}")
    print(f"Shadow pixels: {np.sum(shadow)}")
    print(f"Total mask pixels (undilated): {np.sum(mask)}")
    print(f"Total mask pixels (dilated): {np.sum(mask_dilated)}")

if __name__ == "__main__":
    main()
