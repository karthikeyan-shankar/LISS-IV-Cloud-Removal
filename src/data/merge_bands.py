import os
import rasterio
import numpy as np

def merge_liss4_bands(band2_path, band3_path, band4_path, output_path):
    """
    Stacks separate single-band LISS-IV GeoTIFFs (Green, Red, NIR) 
    into a single 3-channel GeoTIFF.
    
    Order of bands in output:
      - Band 1: Green (BAND2)
      - Band 2: Red (BAND3)
      - Band 3: NIR (BAND4)
    """
    print(f"Starting band merge for: {output_path}")
    print(f"  - Band 2 (Green): {band2_path}")
    print(f"  - Band 3 (Red): {band3_path}")
    print(f"  - Band 4 (NIR): {band4_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open Band 2 first to read the geographic spatial metadata (profile)
    with rasterio.open(band2_path) as b2:
        profile = b2.profile.copy()
        green_data = b2.read(1)
        
    with rasterio.open(band3_path) as b3:
        red_data = b3.read(1)
        
    with rasterio.open(band4_path) as b4:
        nir_data = b4.read(1)
        
    # Update profile for a 3-band output
    profile.update(
        count=3,
        dtype=rasterio.uint16 if green_data.dtype == np.uint16 else rasterio.uint8
    )
    
    # Write the combined image
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(green_data, 1) # Write Green to band 1
        dst.write(red_data, 2)   # Write Red to band 2
        dst.write(nir_data, 3)   # Write NIR to band 3
        
    print(f"Successfully merged LISS-IV bands into: {output_path}\n")

if __name__ == "__main__":
    # Path constants for your downloaded folders
    downloads_dir = r"C:\Users\karth\Downloads"
    
    # 1. Merge Clear/History Scene (February 2026)
    clear_folder = os.path.join(downloads_dir, "RAF06FEB2026047570011000053SSANSTUC00GTDD", "RAF06FEB2026047570011000053SSANSTUC00GTDD")
    merge_liss4_bands(
        band2_path=os.path.join(clear_folder, "BAND2.tif"),
        band3_path=os.path.join(clear_folder, "BAND3.tif"),
        band4_path=os.path.join(clear_folder, "BAND4.tif"),
        output_path=r"c:\Users\karth\OneDrive\Desktop\IS\data\raw\clear\guwahati_clear.tif"
    )
    
    # 2. Merge Cloudy/Today Scene (June 2026)
    cloudy_folder = os.path.join(downloads_dir, "RAF06JUN2026049275011000053SSANSTUC00GTDA", "RAF06JUN2026049275011000053SSANSTUC00GTDA")
    merge_liss4_bands(
        band2_path=os.path.join(cloudy_folder, "BAND2.tif"),
        band3_path=os.path.join(cloudy_folder, "BAND3.tif"),
        band4_path=os.path.join(cloudy_folder, "BAND4.tif"),
        output_path=r"c:\Users\karth\OneDrive\Desktop\IS\data\raw\cloudy\guwahati_cloudy.tif"
    )
