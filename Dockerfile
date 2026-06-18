# Use official Python slim base image
FROM python:3.10-slim

# Install system dependencies needed for OpenCV, Rasterio (GDAL), and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files (including data/raw/clear database and models/)
COPY . .

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Command to start the FastAPI application
CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "7860"]
