import os
import shutil
import numpy as np
import rasterio
import torch
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import sys
import glob

# Add src/models to import pathway
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
try:
    from networks import Generator
except ImportError:
    Generator = None

app = FastAPI(title="ISRO LISS-IV Cloud Removal Portal")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/web_uploads"
STATIC_PREVIEW_DIR = "src/web/static/previews"
CLEAR_DATABASE_DIR = "data/raw/clear" # Folder acting as our historical reference database

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_PREVIEW_DIR, exist_ok=True)
os.makedirs(CLEAR_DATABASE_DIR, exist_ok=True)

# Serve static files for previews
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

def save_preview_png(tif_path, base_filename, ref_limits=None):
    """
    Reads a LISS-IV GeoTIFF (Green, Red, NIR bands) and saves three types of 
    previews: Natural Color, False Color Composite (FCC), and NDVI Heatmap.
    Uses reference limits for consistent contrast stretching across pairs.
    """
    natural_filename = f"natural_{base_filename.replace('.png', '')}.png"
    fcc_filename = f"fcc_{base_filename.replace('.png', '')}.png"
    ndvi_filename = f"ndvi_{base_filename.replace('.png', '')}.png"
    
    natural_path = os.path.join(STATIC_PREVIEW_DIR, natural_filename)
    fcc_path = os.path.join(STATIC_PREVIEW_DIR, fcc_filename)
    ndvi_path = os.path.join(STATIC_PREVIEW_DIR, ndvi_filename)
    
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
        
    limits = []
    stretched_bands = []
    
    # Stretch each band using either calculated or passed limits
    for idx in range(3):
        band = data[idx]
        if ref_limits is not None:
            p1, p99 = ref_limits[idx]
        else:
            p99 = np.percentile(band, 99)
            p1 = np.percentile(band, 1)
            
        limits.append((p1, p99))
        
        if p99 > p1:
            stretched = np.clip((band - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)
        else:
            max_val = band.max() if band.max() > 0 else 1.0
            stretched = np.clip((band / max_val) * 255.0, 0, 255).astype(np.uint8)
        stretched_bands.append(stretched)
            
    g_stretched, r_stretched, nir_stretched = stretched_bands
    
    # 1. Natural Color: R=Red, G=Green, B=(Red+Green)/2
    b_nat = ((r_stretched.astype(np.float16) + g_stretched.astype(np.float16)) / 2).astype(np.uint8)
    natural_bgr = cv2.merge([b_nat, g_stretched, r_stretched])
    cv2.imwrite(natural_path, natural_bgr)
    
    # 2. False Color Composite (FCC): R=NIR, G=Red, B=Green
    fcc_bgr = cv2.merge([g_stretched, r_stretched, nir_stretched])
    cv2.imwrite(fcc_path, fcc_bgr)
    
    # 3. NDVI Heatmap: (NIR - Red) / (NIR + Red)
    red = data[1]
    nir = data[2]
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi_norm = ((ndvi + 1.0) / 2.0 * 255.0).astype(np.uint8)
    ndvi_colored = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_JET)
    cv2.imwrite(ndvi_path, ndvi_colored)
    
    return {
        "previews": {
            "natural": f"/static/previews/{natural_filename}",
            "fcc": f"/static/previews/{fcc_filename}",
            "ndvi": f"/static/previews/{ndvi_filename}"
        },
        "limits": limits
    }

def find_matching_historical_image(cloudy_bounds, crs):
    """
    Searches the historical database folder (data/raw/clear/) for a cloud-free GeoTIFF 
    whose geographic bounding box overlaps with the uploaded cloudy image.
    """
    clear_files = glob.glob(os.path.join(CLEAR_DATABASE_DIR, "*.tif"))
    
    for clear_file in clear_files:
        try:
            with rasterio.open(clear_file) as src:
                ref_bounds = src.bounds
                ref_crs = src.crs
                
                # Verify they are in the same Coordinate Reference System (CRS)
                if ref_crs != crs:
                    continue
                
                # Check for bounding box overlap:
                # If they do not overlap, one must be to the left, right, top, or bottom of the other
                overlap = not (
                    cloudy_bounds.right < ref_bounds.left or
                    cloudy_bounds.left > ref_bounds.right or
                    cloudy_bounds.top < ref_bounds.bottom or
                    cloudy_bounds.bottom > ref_bounds.top
                )
                
                if overlap:
                    print(f"AUTOMATIC MATCH: Found matching historical clear scene: {clear_file}")
                    return clear_file
        except Exception as e:
            print(f"Error checking file {clear_file}: {e}")
            
    return None

# Helper function to run overlapping tiled inference to prevent checkerboard/boundary artifacts
def run_tiled_inference(gen, cloudy_data, ref_data, scale, device, patch_size=256, overlap=32):
    _, h, w = cloudy_data.shape
    
    # Create 2D linear blending weight window
    window_1d = np.ones(patch_size, dtype=np.float32)
    if overlap > 0:
        feather = np.linspace(0, 1, overlap)
        window_1d[:overlap] = feather
        window_1d[-overlap:] = feather[::-1]
    window_2d = np.outer(window_1d, window_1d)
    
    stride = patch_size - overlap
    
    # Pad inputs to handle boundary tiles seamlessly
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride
    
    c_padded = np.pad(cloudy_data, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
    r_padded = np.pad(ref_data, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
    
    h_padded, w_padded = c_padded.shape[1], c_padded.shape[2]
    out_padded = np.zeros_like(c_padded)
    weight_padded = np.zeros((h_padded, w_padded), dtype=np.float32)
    
    for y in range(0, h_padded - patch_size + 1, stride):
        for x in range(0, w_padded - patch_size + 1, stride):
            c_crop = c_padded[:, y:y+patch_size, x:x+patch_size]
            r_crop = r_padded[:, y:y+patch_size, x:x+patch_size]
            
            c_norm = (c_crop / scale) * 2.0 - 1.0
            r_norm = (r_crop / scale) * 2.0 - 1.0
            
            input_tensor = torch.cat([
                torch.from_numpy(c_norm).unsqueeze(0),
                torch.from_numpy(r_norm).unsqueeze(0)
            ], dim=1).to(device)
            
            with torch.no_grad():
                output_tensor = gen(input_tensor).squeeze(0).cpu().numpy()
                
            crop_out = ((output_tensor + 1.0) / 2.0) * scale
            
            for b in range(3):
                out_padded[b, y:y+patch_size, x:x+patch_size] += crop_out[b] * window_2d
            weight_padded[y:y+patch_size, x:x+patch_size] += window_2d
            
    weight_padded = np.maximum(weight_padded, 1e-8)
    for b in range(3):
        out_padded[b] /= weight_padded
        
    return out_padded[:, :h, :w]

# Helper function to detect clouds/shadows and return a dilated, feathered mask
def get_feathered_mask(cloudy_data, ref_data, scale=1023.0):
    c_max = np.percentile(cloudy_data, 99)
    r_max = np.percentile(ref_data, 99)
    if c_max <= 0: c_max = 1.0
    if r_max <= 0: r_max = 1.0
    
    c_norm = cloudy_data / c_max
    r_norm = ref_data / r_max
    
    # 1. Cloud detection: bright in Green (band 0) and Red (band 1)
    cloud = (c_norm[0] > 0.45) & (c_norm[1] > 0.45)
    
    # 2. Shadow detection: where today's image is significantly darker than the history reference in Green
    shadow_diff = (r_norm[0] - c_norm[0] > 0.08)
    
    # Combine
    mask = (cloud | shadow_diff).astype(np.uint8)
    
    # Dilate mask using 11x11 kernel to cover blurry cloud edges and shadow margins
    kernel = np.ones((11, 11), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=3)
    
    # Feather mask using Gaussian blur to ensure seamless transitions
    mask_feathered = cv2.GaussianBlur(mask_dilated.astype(np.float32), (21, 21), 7.0)
    mask_feathered = np.clip(mask_feathered, 0.0, 1.0)
    return mask_feathered

# Helper function to inject reference high-frequency details into reconstructed low-frequencies
def frequency_fusion(reconstructed, reference, ksize=21, sigma=5.0):
    bands, h, w = reconstructed.shape
    fused = np.zeros_like(reconstructed)
    
    for b in range(bands):
        recon_band = reconstructed[b]
        ref_band = reference[b]
        
        # Low frequency component of reconstructed (today's translated color)
        recon_low = cv2.GaussianBlur(recon_band, (ksize, ksize), sigma)
        
        # Low frequency component of reference
        ref_low = cv2.GaussianBlur(ref_band, (ksize, ksize), sigma)
        
        # High frequency component of reference (sharp spatial features)
        ref_high = ref_band - ref_low
        
        # Fuse low frequency color with high frequency details
        fused[b] = recon_low + ref_high
        
    return fused

# Helper function to compute NDVI metrics
def get_ndvi_stats(data):
    if data.max() > 1.0:
        data = data / 255.0
    red = data[1]
    nir = data[2]
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return {
        "mean": float(np.mean(ndvi)),
        "max": float(np.max(ndvi)),
        "min": float(np.min(ndvi)),
        "healthy_ratio": float(np.mean(ndvi > 0.3))
    }

@app.post("/api/reconstruct")
async def reconstruct_image(
    cloudy_file: UploadFile = File(...)
):
    """
    Automated endpoint: Receives ONLY the cloudy image.
    Reads its geospatial bounding box, automatically matches it to a historical image,
    runs the GAN, and returns the reconstructed clean GeoTIFF.
    """
    cloudy_path = os.path.join(UPLOAD_DIR, f"cloudy_{cloudy_file.filename}")
    out_path = os.path.join(UPLOAD_DIR, f"reconstructed_{cloudy_file.filename}")
    
    # Save uploaded file
    with open(cloudy_path, "wb") as f:
        shutil.copyfileobj(cloudy_file.file, f)
        
    try:
        # 1. Read input GeoTIFF coordinates
        with rasterio.open(cloudy_path) as src:
            cloudy_data = src.read().astype(np.float32)
            profile = src.profile
            cloudy_bounds = src.bounds
            crs = src.crs
            
        # 2. Automated Database Lookup
        ref_path = find_matching_historical_image(cloudy_bounds, crs)
        
        if not ref_path:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Geospatial Database Notice: No overlapping historical cloud-free reference found in the local database "
                    f"for coordinates [{cloudy_bounds.left:.1f}, {cloudy_bounds.bottom:.1f}, {cloudy_bounds.right:.1f}, {cloudy_bounds.top:.1f}] (CRS: {crs}). "
                    f"Please load the coordinate-registered historical reference file (.tif) into 'data/raw/clear/' to enable automated reconstruction."
                )
            )
                
        # 3. Read matching historical file (cropped to the exact same geo-coordinates)
        with rasterio.open(ref_path) as ref_src:
            from rasterio.windows import from_bounds
            ref_window = from_bounds(
                cloudy_bounds.left, cloudy_bounds.bottom, 
                cloudy_bounds.right, cloudy_bounds.top, 
                ref_src.transform
            ).round()
            
            # Read matching spatial window, padding with 0 if slightly outside bounds
            ref_data = ref_src.read(window=ref_window, boundless=True, fill_value=0).astype(np.float32)
            
        # Align reference dimensions to cloudy dimensions if they differ slightly
        if ref_data.shape[1] != cloudy_data.shape[1] or ref_data.shape[2] != cloudy_data.shape[2]:
            ref_data = ref_data[:, :cloudy_data.shape[1], :cloudy_data.shape[2]]
            
        # 4. Model Inference
        # Find the latest generator checkpoint in models/
        checkpoints = glob.glob("models/generator_epoch_*.pth")
        model_weight_path = None
        if checkpoints:
            try:
                checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("generator_epoch_", "").replace(".pth", "")))
                model_weight_path = checkpoints[-1]
                print(f"AUTOMATIC MODEL LOADING: Loading latest checkpoint: {model_weight_path}")
            except Exception as e:
                model_weight_path = checkpoints[-1]
                print(f"Error sorting checkpoints: {e}. Loading fallback: {model_weight_path}")
        else:
            model_weight_path = "models/generator_epoch_30.pth"
            
        # LISS-IV is 10-bit data (0-1023) stored in 16-bit files.
        # We normalize using a constant scale of 1023.0 to match dataset.py.
        scale = 1023.0
        
        if Generator is not None and model_weight_path and os.path.exists(model_weight_path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gen = Generator(in_channels=6, out_channels=3)
            gen.load_state_dict(torch.load(model_weight_path, map_location=device))
            gen.to(device)
            gen.eval()
            
            # 1. Run tiled inference for seasonal color translation
            tiled_out = run_tiled_inference(gen, cloudy_data, ref_data, scale, device, patch_size=256, overlap=32)
            tiled_out = np.clip(tiled_out, 0, scale)
            
            # 2. Run frequency fusion to inject reference high-frequency textures into generator predictions
            fused_out = frequency_fusion(tiled_out, ref_data, ksize=21, sigma=5.0)
            fused_out = np.clip(fused_out, 0, scale)
            
            # 3. Detect clouds & shadows, and create a feathered blend mask
            blend_mask = get_feathered_mask(cloudy_data, ref_data, scale=scale)
            
            # 4. Blend: keep original sharp pixels in clear sky, use fused sharp/seasonal pixels in cloudy regions
            reconstructed_data = np.zeros_like(cloudy_data)
            for b in range(3):
                reconstructed_data[b] = (1.0 - blend_mask) * cloudy_data[b] + blend_mask * fused_out[b]
                
            reconstructed_data = np.clip(reconstructed_data, 0, scale).astype(profile['dtype'])
        else:
            # Fallback blending mode (scale-independent)
            c_max = cloudy_data.max() if cloudy_data.max() > 0 else 1.0
            c_normalized = cloudy_data / c_max
            cloud_mask = (c_normalized[0] > 0.6) & (c_normalized[1] > 0.6)
            
            reconstructed_data = np.zeros_like(cloudy_data)
            for b in range(cloudy_data.shape[0]):
                reconstructed_data[b] = np.where(cloud_mask, ref_data[b], cloudy_data[b])
            reconstructed_data = reconstructed_data.astype(profile['dtype'])
            
        # Save output GeoTIFF
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(reconstructed_data)
            
        # Generate browser-renderable PNG previews (Natural, FCC, and NDVI Heatmaps)
        # Use consistent limits from reconstructed output to prevent dynamic range skew between previews
        out_res = save_preview_png(out_path, "reconstructed_preview.png")
        cloudy_res = save_preview_png(cloudy_path, "cloudy_preview.png", ref_limits=out_res["limits"])
        
        # Calculate NDVI stats
        cloudy_ndvi = get_ndvi_stats(cloudy_data)
        reconstructed_ndvi = get_ndvi_stats(reconstructed_data)
        
        return {
            "status": "success",
            "cloudy_preview": cloudy_res["previews"]["natural"],
            "reconstructed_preview": out_res["previews"]["natural"],
            "cloudy_previews": cloudy_res["previews"],
            "reconstructed_previews": out_res["previews"],
            "download_url": f"/api/download/{os.path.basename(out_path)}",
            "matched_reference": os.path.basename(ref_path),
            "metrics": {
                "cloudy_ndvi": cloudy_ndvi,
                "reconstructed_ndvi": reconstructed_ndvi,
                "improvement": f"{float(reconstructed_ndvi['healthy_ratio'] - cloudy_ndvi['healthy_ratio']) * 100:.1f}%"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/tiff", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

# Serve the index.html page
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = "src/web/templates/index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: index.html not found under src/web/templates/</h3>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
