import os
import rasterio
from rasterio.windows import Window
import numpy as np

def load_geotiff(file_path):
    """
    Reads a GeoTIFF file and returns the pixel data as a NumPy array 
    and its geographic profile (coordinate metadata).
    """
    with rasterio.open(file_path) as src:
        # Read all bands (Green, Red, NIR)
        # Shape: (bands, height, width)
        data = src.read()
        profile = src.profile
    return data, profile

def save_geotiff(file_path, data, profile):
    """
    Saves a NumPy array as a GeoTIFF file using the provided geographic profile.
    """
    # Ensure the data matches the profile specifications
    profile.update(
        dtype=data.dtype,
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2]
    )
    with rasterio.open(file_path, 'w', **profile) as dst:
        dst.write(data)

def slice_into_patches(image_path, output_dir, patch_size=256, stride=256):
    """
    Slices a large GeoTIFF image into smaller square patches (tiles).
    Crucially, it updates the geographic metadata (geotransform) for each patch
    so that the patches remain georeferenced.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        height = src.height
        width = src.width
        
        # Loop through the image with sliding window
        patch_count = 0
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # Define the cropping window
                window = Window(x, y, patch_size, patch_size)
                
                # Read the window data
                patch_data = src.read(window=window)
                
                # Check if patch is mostly empty (e.g. edge of swath filled with nodata)
                # If more than 50% of the pixels are zero, skip this patch
                if np.mean(patch_data == 0) > 0.5:
                    continue
                
                # Calculate new geographic transform coordinates for this specific patch
                patch_transform = rasterio.windows.transform(window, src.transform)
                
                # Update metadata profile for the patch
                patch_meta = meta.copy()
                patch_meta.update({
                    'height': patch_size,
                    'width': patch_size,
                    'transform': patch_transform
                })
                
                # Save the patch as a separate GeoTIFF
                patch_name = f"{os.path.basename(image_path).replace('.tif', '')}_patch_{patch_count}.tif"
                patch_path = os.path.join(output_dir, patch_name)
                
                with rasterio.open(patch_path, 'w', **patch_meta) as dst:
                    dst.write(patch_data)
                
                patch_count += 1
                
        print(f"Successfully sliced {image_path} into {patch_count} patches in: {output_dir}")

if __name__ == "__main__":
    print("Geospatial preprocessing module initialized!")
    # Example usage:
    # slice_into_patches("data/raw/cloudy/scene1.tif", "data/patches/cloudy")
