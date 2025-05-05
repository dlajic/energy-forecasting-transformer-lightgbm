import torch
import importlib
import subprocess
import sys

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def check_device():
    # **Check for NVIDIA GPU (CUDA)**
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use NVIDIA GPU
        backend = "CUDA (NVIDIA)"
        mixed_precision = True  # Use Automatic Mixed Precision (AMP)

    # **If no NVIDIA GPU, check for AMD GPU (DirectML) only in Windows**
    else:
        try:
            # Only try DirectML if the environment is Windows and DirectML is installed
            if "win32" in sys.platform:
                torch_directml = importlib.import_module("torch_directml")
                if torch_directml.device_count() > 0:
                    device = torch_directml.device()  # Use AMD GPU with DirectML
                    backend = "DirectML (AMD)"
                    mixed_precision = False  # No AMP for AMD GPU
                else:
                    raise ImportError  # AMD GPU not found
            else:
                device = torch.device("cpu")
                backend = "CPU"
                mixed_precision = False  # No AMP for CPU

        except ImportError:
            # If DirectML is not installed or AMD GPU not found
            device = torch.device("cpu")
            backend = "CPU"
            mixed_precision = False  # No AMP for CPU

    # Print the chosen device info
    print(f"Training is running on: {backend} ({device})")

    # **Initialize scaler (only for NVIDIA)**
    if mixed_precision:
        scaler = torch.amp.GradScaler()
    else:
        scaler = None  # No scaler needed for AMD/CPU

    return device, backend, scaler

if __name__ == "__main__":
    device, backend, scaler = check_device()
