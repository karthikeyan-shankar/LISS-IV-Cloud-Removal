import os
import sys

# Add src/data to imports path
sys.path.append(os.path.dirname(__file__))
from preprocess import slice_into_patches

if __name__ == "__main__":
    print("Starting GeoTIFF patching pipeline...")
    
    # 1. Slice Clear Guwahati Scene
    slice_into_patches(
        image_path=r"data/raw/clear/guwahati_clear.tif",
        output_dir=r"data/patches/clear",
        patch_size=256,
        stride=256
    )
    
    # 2. Slice Cloudy Guwahati Scene
    slice_into_patches(
        image_path=r"data/raw/cloudy/guwahati_cloudy.tif",
        output_dir=r"data/patches/cloudy",
        patch_size=256,
        stride=256
    )
    
    print("Patching complete!")
