#!/bin/bash
# Installation script for UNet with EfficientNet-B7

echo "🚀 Installing UNet with EfficientNet-B7"
echo "======================================"

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python $python_version detected, but $required_version or higher is required"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip first."
    exit 1
fi

echo "✅ pip3 found"

# Ask user for installation type
echo ""
echo "Select installation type:"
echo "1) Minimal (core dependencies only)"
echo "2) Full (with development tools)"
echo "3) GPU (with CUDA support)"
echo "4) Custom (manual selection)"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "📦 Installing minimal dependencies..."
        pip3 install -r requirements-minimal.txt
        ;;
    2)
        echo "📦 Installing full dependencies..."
        pip3 install -r requirements.txt
        ;;
    3)
        echo "📦 Installing GPU dependencies..."
        pip3 install -r requirements-gpu.txt
        ;;
    4)
        echo "📦 Installing custom dependencies..."
        echo "Available requirements files:"
        echo "- requirements-minimal.txt (core only)"
        echo "- requirements.txt (full)"
        echo "- requirements-gpu.txt (GPU optimized)"
        read -p "Enter requirements file name: " req_file
        pip3 install -r "$req_file"
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

# Verify installation
echo ""
echo "🔍 Verifying installation..."

python3 -c "
import sys
try:
    import tensorflow as tf
    print(f'✅ TensorFlow {tf.__version__} installed')
except ImportError:
    print('❌ TensorFlow not installed')
    sys.exit(1)

try:
    import cv2
    print(f'✅ OpenCV {cv2.__version__} installed')
except ImportError:
    print('❌ OpenCV not installed')
    sys.exit(1)

try:
    import albumentations
    print(f'✅ Albumentations {albumentations.__version__} installed')
except ImportError:
    print('❌ Albumentations not installed')
    sys.exit(1)

try:
    import numpy
    print(f'✅ NumPy {numpy.__version__} installed')
except ImportError:
    print('❌ NumPy not installed')
    sys.exit(1)

print('✅ All core dependencies verified!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Prepare your dataset"
    echo "2. Run: python train.py --help"
    echo "3. Or use: ./train_severstal.sh for Severstal dataset"
else
    echo ""
    echo "❌ Installation verification failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi
