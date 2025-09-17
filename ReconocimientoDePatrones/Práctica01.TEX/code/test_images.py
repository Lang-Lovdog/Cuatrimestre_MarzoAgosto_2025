"""
Quick test script for validating image loading
"""

import sys                                         #type: ignore
import cv2                                         #type: ignore
import os                                          #type: ignore
import numpy     as     np                         #type: ignore
from   pathlib   import Path                       #type: ignore
sys.path.append(str(Path(__file__).parent))

from LovdogDF    import LovdogDataFrames           #type: ignore
from LovdogSDH   import SDHFeatures                #type: ignore
from LovdogData  import visualize_sample_images    #type: ignore
from LovdogData  import load_image_file            #type: ignore

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    packages = [
        'cv2', 'numpy', 'pandas', 'sklearn', 
        'skimage', 'matplotlib', 'tqdm'
    ]
    
    all_imports_ok = True
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def test_opencv():
    """Test OpenCV installation."""
    print("\nTesting OpenCV...")
    try:
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (10, 10), (90, 90), (255, 0, 0), 2)
        print("✓ OpenCV basic functionality working")
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_image_loading():
    """Test image loading functionality."""
    print("\nTesting image loading...")
    
    try:
        # Test single image loading
        test_image_path = None
        for root, dirs, files in os.walk('res/imagenes'):
            for file in files:
                if file.lower().endswith('.png'):
                    test_image_path = os.path.join(root, file)
                    break
            if test_image_path:
                break
        
        if test_image_path and os.path.exists(test_image_path):
            image = load_image_file(test_image_path)
            if image and len(image) > 0:
                print(f"✓ Image loading successful: {image[0].shape}")
                return True
            else:
                print("❌ Image loading failed")
                return False
        else:
            print("⚠️  No PNG images found for testing")
            return True
            
    except Exception as e:
        print(f"❌ Image loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("LOVDOG TOOLKIT - SETUP VALIDATION")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_opencv()
    success &= test_image_loading()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ SETUP VALIDATION PASSED!")
        print("You can now run: python res/main.py")
    else:
        print("❌ SETUP VALIDATION FAILED")
        print("Please check the errors above and run setup.sh again")
        sys.exit(1)
    print("=" * 50)
