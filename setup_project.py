import os

dirs = [
    "data/raw/cloudy",
    "data/raw/clear",
    "data/patches/cloudy",
    "data/patches/clear",
    "src/data",
    "src/models",
    "src/web/static",
    "src/web/templates"
]

for d in dirs:
    path = os.path.join(os.getcwd(), d)
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")

# Create empty __init__.py files for imports
open("src/__init__.py", "w").close()
open("src/data/__init__.py", "w").close()
open("src/models/__init__.py", "w").close()

print("Project directories initialized successfully!")
