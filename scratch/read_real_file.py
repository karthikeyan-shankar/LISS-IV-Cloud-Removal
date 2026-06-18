import rasterio
import numpy as np

def main():
    file_path = "data/web_uploads/reconstructed_guwahati_cloudy_test.tif"
    
    print(f"Opening reconstructed GeoTIFF: {file_path}")
    with rasterio.open(file_path) as src:
        print(f"Driver Name:            {src.driver}")
        print(f"Width x Height:        {src.width} x {src.height} pixels")
        print(f"Band Count:            {src.count}")
        print(f"CRS (Projection):      {src.crs}")
        
        # Read the bands
        green = src.read(1)
        red = src.read(2)
        nir = src.read(3)
        
        print("\n--- BAND REFLECTANCE STATISTICS ---")
        print(f"Band 1 (Green): Min={green.min()}, Max={green.max()}, Mean={green.mean():.2f}")
        print(f"Band 2 (Red):   Min={red.min()},   Max={red.max()},   Mean={red.mean():.2f}")
        print(f"Band 3 (NIR):   Min={nir.min()},   Max={nir.max()},   Mean={nir.mean():.2f}")
        
        # Calculate NDVI stats
        # Convert to float for safe division
        red_f = red.astype(np.float32)
        nir_f = nir.astype(np.float32)
        ndvi = (nir_f - red_f) / (nir_f + red_f + 1e-8)
        
        print("\n--- VEGETATION INDEX (NDVI) METRICS ---")
        print(f"NDVI Range:     [{ndvi.min():.4f}, {ndvi.max():.4f}]")
        print(f"NDVI Mean:      {ndvi.mean():.4f}")
        print(f"Healthy Canopy Ratio (NDVI > 0.3): {np.mean(ndvi > 0.3) * 100:.2f}%")

if __name__ == "__main__":
    main()
