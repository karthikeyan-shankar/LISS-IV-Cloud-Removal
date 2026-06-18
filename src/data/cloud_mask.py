import numpy as np
import rasterio

def generate_simple_cloud_mask(image_data, threshold=0.6):
    """
    Generates a binary cloud mask from LISS-IV band data (Green, Red, NIR).
    
    Parameters:
    - image_data: NumPy array of shape (bands, height, width)
      Expected bands: index 0 = Green, index 1 = Red, index 2 = NIR.
      Assumes pixel values are normalized to 0.0 - 1.0 (Surface Reflectance).
    
    Returns:
    - mask: Binary NumPy array of shape (1, height, width) where 1=Cloud, 0=Clear.
    """
    # LISS-IV bands:
    green = image_data[0]
    red = image_data[1]
    
    # Clouds are highly reflective in both visible bands (Green and Red)
    # If both bands are brighter than the threshold, we classify it as a cloud.
    cloud_pixels = (green > threshold) & (red > threshold)
    
    # Convert boolean to binary 0/1 array
    mask = np.expand_dims(cloud_pixels.astype(np.uint8), axis=0)
    return mask

def create_mask_for_file(image_path, output_mask_path, threshold=0.6):
    """
    Loads a GeoTIFF image, creates a cloud mask, and saves it as a new GeoTIFF.
    """
    with rasterio.open(image_path) as src:
        data = src.read()
        profile = src.profile
        
        # If raw DN values are 0-255 instead of normalized 0-1, 
        # we adjust the threshold dynamically.
        if data.max() > 1.0:
            norm_data = data / 255.0
        else:
            norm_data = data
            
        mask = generate_simple_cloud_mask(norm_data, threshold=threshold)
        
        # Update metadata profile for single-band binary mask
        profile.update(
            dtype=rasterio.uint8,
            count=1
        )
        
        with rasterio.open(output_mask_path, 'w', **profile) as dst:
            dst.write(mask)
            
    print(f"Cloud mask created and saved to: {output_mask_path}")

if __name__ == "__main__":
    print("Cloud masking module initialized!")
