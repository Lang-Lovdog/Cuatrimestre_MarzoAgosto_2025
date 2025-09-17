#!/bin/bash

# LovdogToolkit Setup Script
echo "=========================================="
echo "   LOVDOG TOOLKIT SETUP SCRIPT"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✓ All requirements installed successfully."
    else
        echo "❌ Failed to install some requirements."
        echo "Trying with alternative package sources..."
        pip install --upgrade pip setuptools wheel
        pip install -r requirements.txt --extra-index-url https://pypi.org/simple/
    fi
else
    echo "❌ requirements.txt not found in  directory."
    echo "Installing default packages..."
    pip install opencv-python scikit-image scikit-learn pandas numpy matplotlib
fi

# Test basic imports
echo "Testing basic imports..."
python -c "
try:
    import cv2
    import numpy as np
    import pandas as pd
    from skimage import io, color
    from sklearn import datasets
    print('✓ All core imports successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

# Create necessary directories
echo "Creating output directories..."
mkdir -p results features models plots

echo "=========================================="
echo "   SETUP COMPLETED SUCCESSFULLY! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Test the setup:"
echo "   python quick_test.py"
echo ""
echo "3. Run the full workflow:"
echo "   python main.py"
echo ""
echo "4. Deactivate when done:"
echo "   deactivate"
echo "=========================================="
