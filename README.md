# ISRO BAH 2026: Cloud Removal and Surface Reconstruction for LISS-IV Imagery

This project is a Generative AI-based framework developed for the **Bharatiya Antariksh Hackathon (BAH) 2026** (Problem Statement 2). It utilizes a Hybrid Generative Adversarial Network (GAN) to reconstruct cloud-covered regions in high-resolution LISS-IV satellite imagery by fusing today's cloudy scenes with historical cloud-free reference scenes.

---

## 📁 Project Structure

*   `data/`
    *   `raw/` - Store raw GeoTIFF scenes from Bhoonidhi (`cloudy/` and `clear/`)
    *   `patches/` - Patches sliced into $256 \times 256$ tiles by our preprocessing pipeline
*   `src/`
    *   `data/` - Preprocessing, patching, and cloud masking scripts
    *   `models/` - PyTorch GAN architectures, custom loss functions, and training loops
    *   `web/` - FastAPI backend and React frontend dashboard for live visualization

---

## 🛠️ Tech Stack & Requirements

1.  **Programming Language:** Python 3.10+
2.  **AI & Vision:** PyTorch, torchvision, OpenCV, scikit-image, albumentations
3.  **Geospatial Processing:** Rasterio, NumPy (for direct GeoTIFF manipulation)
4.  **UI & Backend:** FastAPI, Uvicorn, React, Tailwind CSS

To install dependencies, run:
```bash
pip install -r requirements.txt
```

---

## 👥 Team Workflow

1.  **Dataset Sourcing:**
    *   Find matching LISS-IV scenes of the same coordinates (e.g., Guwahati or Shillong) on ISRO's Bhoonidhi portal.
    *   Download one cloudy scene and one historical clear scene, and place them in `data/raw/cloudy` and `data/raw/clear`.
2.  **Dataset Preparation:**
    *   Run the preprocessing scripts to slice the scenes into coordinate-registered $256 \times 256$ tiles.
3.  **Model Training:**
    *   Train the Pix2Pix GAN model using the custom PyTorch training scripts.
4.  **Operational Interface:**
    *   Run the FastAPI local server to launch the visual comparison portal.
