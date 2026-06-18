import requests
import os

def test_api():
    url = "http://127.0.0.1:8000/api/reconstruct"
    file_path = "chennai_cloudy_test.tif"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
        
    print(f"Sending POST request to {url} with file {file_path}...")
    
    with open(file_path, "rb") as f:
        files = {"cloudy_file": (os.path.basename(file_path), f, "image/tiff")}
        try:
            response = requests.post(url, files=files, timeout=60)
            print(f"Response Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n=== SUCCESS: API TEST PASSED ===")
                print(f"Status:                 {result['status']}")
                print(f"Matched Reference:      {result['matched_reference']}")
                print(f"NDVI Reconstructed Mean: {result['metrics']['reconstructed_ndvi']['mean']:.4f}")
                print(f"NDVI Improvement:       {result['metrics']['improvement']}")
                print(f"Download URL:           {result['download_url']}")
                print("\nPreviews Generated:")
                print(f"Natural:                {result['reconstructed_previews']['natural']}")
                print(f"FCC:                    {result['reconstructed_previews']['fcc']}")
                print(f"NDVI:                   {result['reconstructed_previews']['ndvi']}")
            else:
                print(f"Error Response: {response.text}")
        except Exception as e:
            print(f"Failed to connect to the server: {e}")

if __name__ == "__main__":
    test_api()
