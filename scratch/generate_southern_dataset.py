import numpy as np
import rasterio
from rasterio.transform import from_origin
import os

def generate_scene():
    print("Generating simulated South Indian (Chennai) dataset...")
    
    # Dimensions
    h, w = 1024, 1024
    
    # Coordinate system: EPSG:32644 (UTM Zone 44N - covers Southern India / Chennai)
    # Origin coordinates in meters (e.g., Chennai area)
    x_origin = 400000.0
    y_origin = 1440000.0
    pixel_size = 5.0 # 5-meter resolution for LISS-IV
    
    transform = from_origin(x_origin, y_origin, pixel_size, pixel_size)
    
    # Create clear background: Red soil & agricultural fields
    # Band 1: Green, Band 2: Red, Band 3: NIR
    # Soil: Red soil is bright in Red (band 2), low in Green and NIR
    soil_green = np.random.normal(60, 3, (h, w))
    soil_red = np.random.normal(120, 5, (h, w)) # High red for iron-rich red soil
    soil_nir = np.random.normal(80, 4, (h, w))  # Low NIR for bare soil
    
    # Create square agricultural fields
    # Generate grid lines for fields
    grid_y, grid_x = np.indices((h, w))
    field_size = 128
    fields_mask = ((grid_y // field_size) + (grid_x // field_size)) % 2 == 0
    
    # Vegetation fields: High NIR, Low Red, Moderate Green
    veg_green = np.random.normal(85, 4, (h, w))
    veg_red = np.random.normal(45, 3, (h, w))
    veg_nir = np.random.normal(240, 10, (h, w)) # High NIR for healthy crops
    
    green = np.where(fields_mask, veg_green, soil_green)
    red = np.where(fields_mask, veg_red, soil_red)
    nir = np.where(fields_mask, veg_nir, soil_nir)
    
    clear_data = np.stack([green, red, nir]).astype(np.uint16)
    
    # Create cloudy image (copy clear and draw clouds/shadows)
    cloudy_green = green.copy()
    cloudy_red = red.copy()
    cloudy_nir = nir.copy()
    
    # Draw cloud blob
    cloud_mask = np.zeros((h, w), dtype=np.float32)
    # Center-left cloud
    ccy, ccx = h // 3, w // 3
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((x_idx - ccx)**2 + (y_idx - ccy)**2)
    blob = np.clip(1.0 - (dist / 140.0), 0, 1)
    # Smoothstep
    blob = blob * blob * (3.0 - 2.0 * blob)
    cloud_mask = np.maximum(cloud_mask, blob)
    
    # Draw bright clouds (scale up to ~800, bright in all bands)
    cloud_intensity = 650.0
    for b_data, bg_val in zip([cloudy_green, cloudy_red, cloudy_nir], [green, red, nir]):
        # Blend white cloud body
        b_data[:] = (cloud_mask * cloud_intensity + (1.0 - cloud_mask) * b_data).astype(np.uint16)
        
    # Draw corresponding shadow (shifted bottom-right)
    shadow_mask = np.roll(cloud_mask, shift=(40, 40), axis=(0, 1))
    # Dim the shadowed pixels by 45%
    cloudy_green = np.where(shadow_mask > 0.2, cloudy_green * (1.0 - 0.45 * shadow_mask), cloudy_green).astype(np.uint16)
    cloudy_red = np.where(shadow_mask > 0.2, cloudy_red * (1.0 - 0.45 * shadow_mask), cloudy_red).astype(np.uint16)
    cloudy_nir = np.where(shadow_mask > 0.2, cloudy_nir * (1.0 - 0.45 * shadow_mask), cloudy_nir).astype(np.uint16)
    
    cloudy_data = np.stack([cloudy_green, cloudy_red, cloudy_nir]).astype(np.uint16)
    
    # Profile metadata
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'nodata': None,
        'width': w,
        'height': h,
        'count': 3,
        'crs': rasterio.crs.CRS.from_epsg(32644), # UTM Zone 44N (Chennai/Southern India)
        'transform': transform
    }
    
    # Save clear reference in database
    clear_out_dir = "data/raw/clear"
    os.makedirs(clear_out_dir, exist_ok=True)
    clear_path = os.path.join(clear_out_dir, "chennai_clear.tif")
    with rasterio.open(clear_path, 'w', **profile) as dst:
        dst.write(clear_data)
    print(f"Saved: {clear_path}")
        
    # Save cloudy scene in root
    cloudy_path = "chennai_cloudy_test.tif"
    with rasterio.open(cloudy_path, 'w', **profile) as dst:
        dst.write(cloudy_data)
    print(f"Saved: {cloudy_path}")
    
    print("Simulated Chennai dataset generated successfully!")

if __name__ == "__main__":
    generate_scene()
