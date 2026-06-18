import os
import rasterio
from rasterio.windows import from_bounds

cloudy_path = "guwahati_cloudy_test.tif"
large_clear_path = "data/raw/clear/guwahati_clear.tif"

if not os.path.exists(cloudy_path):
    print(f"Error: {cloudy_path} not found!")
    exit(1)

if not os.path.exists(large_clear_path):
    print(f"Error: {large_clear_path} not found!")
    exit(1)

print(f"Reading cloudy scene bounds from: {cloudy_path}")
with rasterio.open(cloudy_path) as src_cloudy:
    cloudy_bounds = src_cloudy.bounds
    cloudy_crs = src_cloudy.crs
    print(f"Cloudy bounds: {cloudy_bounds}")
    print(f"Cloudy CRS: {cloudy_crs}")

print(f"Reading clear scene from: {large_clear_path}")
with rasterio.open(large_clear_path) as src_clear:
    clear_crs = src_clear.crs
    print(f"Clear CRS: {clear_crs}")
    
    # Get window for cloudy bounds
    window = from_bounds(
        cloudy_bounds.left, cloudy_bounds.bottom,
        cloudy_bounds.right, cloudy_bounds.top,
        src_clear.transform
    ).round()
    
    print(f"Window: {window}")
    
    # Read cropped data
    data = src_clear.read(window=window, boundless=True, fill_value=0)
    
    # Get transform of the cropped window
    cropped_transform = src_clear.window_transform(window)

# Build profile for cropped clear image
profile = {
    'driver': 'GTiff',
    'dtype': data.dtype,
    'nodata': None,
    'width': int(window.width),
    'height': int(window.height),
    'count': int(data.shape[0]),
    'crs': clear_crs,
    'transform': cropped_transform
}

# Temporarily write to a temporary file, then replace the original large file
temp_clear_path = "data/raw/clear/guwahati_clear_temp.tif"
print(f"Saving cropped clear file to: {temp_clear_path}")
with rasterio.open(temp_clear_path, 'w', **profile) as dst:
    dst.write(data)

# Close and replace
if os.path.exists(temp_clear_path):
    os.remove(large_clear_path)
    os.rename(temp_clear_path, large_clear_path)
    print("Overwrote guwahati_clear.tif successfully!")
