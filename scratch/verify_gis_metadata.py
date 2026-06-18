import rasterio
import os

def verify_metadata():
    input_path = "guwahati_cloudy_test.tif"
    output_path = "data/web_uploads/reconstructed_guwahati_cloudy_test.tif"
    
    if not os.path.exists(output_path):
        print(f"Error: {output_path} does not exist. Please run a reconstruction first.")
        return
        
    print("=== GEOSPATIAL METADATA VERIFICATION ===")
    
    with rasterio.open(input_path) as src_in:
        in_crs = src_in.crs
        in_transform = src_in.transform
        in_bounds = src_in.bounds
        in_shape = src_in.shape
        in_bands = src_in.count
        in_dtype = src_in.dtypes[0]
        
    with rasterio.open(output_path) as src_out:
        out_crs = src_out.crs
        out_transform = src_out.transform
        out_bounds = src_out.bounds
        out_shape = src_out.shape
        out_bands = src_out.count
        out_dtype = src_out.dtypes[0]
        
    print(f"Property            | Input Scene (Cloudy)     | Output Scene (Reconstructed)")
    print(f"-----------------------------------------------------------------------------")
    print(f"CRS                 | {in_crs}               | {out_crs}")
    print(f"Dimensions (H, W)   | {in_shape}               | {out_shape}")
    print(f"Number of Bands     | {in_bands}                        | {out_bands}")
    print(f"Data Type           | {in_dtype}                   | {out_dtype}")
    print(f"Bounds (Left)       | {in_bounds.left:.3f}              | {out_bounds.left:.3f}")
    print(f"Bounds (Right)      | {in_bounds.right:.3f}             | {out_bounds.right:.3f}")
    print(f"Bounds (Top)        | {in_bounds.top:.3f}             | {out_bounds.top:.3f}")
    print(f"Bounds (Bottom)     | {in_bounds.bottom:.3f}            | {out_bounds.bottom:.3f}")
    
    # Check alignment matching
    aligns_crs = (in_crs == out_crs)
    aligns_transform = (in_transform == out_transform)
    aligns_bounds = (in_bounds == out_bounds)
    aligns_shape = (in_shape == out_shape)
    
    print("\n=== VERIFICATION RESULTS ===")
    print(f"1. CRS Match:                {'PASSED' if aligns_crs else 'FAILED'}")
    print(f"2. Transform Matrix Match:   {'PASSED' if aligns_transform else 'FAILED'}")
    print(f"3. Bounding Box Match:       {'PASSED' if aligns_bounds else 'FAILED'}")
    print(f"4. Spatial Dimensions Match: {'PASSED' if aligns_shape else 'FAILED'}")
    
    if aligns_crs and aligns_transform and aligns_bounds and aligns_shape:
        print("\nSUCCESS: The output GeoTIFF is 100% georeferenced and matches the input scene perfectly!")
        print("When imported into GIS software (QGIS / ArcGIS), it will overlay exactly on top of the original telemetry.")
    else:
        print("\nWARNING: Geospatial metadata mismatch detected.")

if __name__ == "__main__":
    verify_metadata()
