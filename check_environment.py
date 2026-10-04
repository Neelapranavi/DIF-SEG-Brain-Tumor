import sys
import os

print("========================================")
print("DIF-SEG ENVIRONMENT CHECK")
print("========================================\n")

# 1. Check Python Version
print(f"[CHECK] Python Version: {sys.version.split(' ')[0]}")
if sys.version_info < (3, 7):
    print("  -> [WARNING] Python 3.7+ is recommended.")
else:
    print("  -> [OK]")

# 2. Check Required Packages
required_packages = {
    'torch': 'PyTorch',
    'streamlit': 'Streamlit',
    'numpy': 'NumPy',
    'cv2': 'OpenCV',
    'PIL': 'Pillow',
    'scipy': 'SciPy',
    'plotly': 'Plotly',
    'reportlab': 'ReportLab'
}

print("\n[CHECK] Dependencies:")
for mod, name in required_packages.items():
    try:
        __import__(mod)
        print(f"  -> [OK] {name} is installed.")
    except ImportError:
        print(f"  -> [ERROR] {name} is missing!")

# 3. Check PyTorch CUDA Availability
print("\n[CHECK] Hardware Acceleration:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  -> [OK] CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  -> [INFO] CUDA is not available. CPU will be used.")
except ImportError:
    print("  -> [ERROR] PyTorch not installed, cannot check CUDA.")

# 4. Check Required Directories & Files
print("\n[CHECK] Project Structure:")
required_paths = [
    "app.py",
    "config.yaml",
    "requirements.txt",
    "models/hub_model.pth",
    "src",
    "src/model/inference.py"
]

for path in required_paths:
    if os.path.exists(path):
        print(f"  -> [OK] Found: {path}")
    else:
        print(f"  -> [ERROR] Missing: {path}")

print("\n========================================")
print("Environment check completed.")
print("========================================")
