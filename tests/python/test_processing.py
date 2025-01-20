import pytest
import numpy as np
import os
from pathlib import Path
import fast_image_processing as fip
from fast_image_processing import Hardware

# Get the directory containing test images
TEST_DIR = Path(__file__).parent
RGB_FILENAME = "test_rgb.jpg"
GRAY_FILENAME = "test_gray.png"

@pytest.fixture(scope="session")
def rgb_image():
    """Load RGB test image from file."""
    image_path = TEST_DIR / RGB_FILENAME
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")
    return fip.Image(str(image_path))

@pytest.fixture(scope="session")
def gray_image():
    """Load grayscale test image from file."""
    image_path = TEST_DIR / GRAY_FILENAME
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")
    return fip.Image(str(image_path))

def test_images_exist():
    """Verify test images are present and valid."""
    rgb_path = TEST_DIR / RGB_FILENAME
    gray_path = TEST_DIR / GRAY_FILENAME
    
    assert rgb_path.exists(), f"RGB test image not found: {rgb_path}"
    assert gray_path.exists(), f"Grayscale test image not found: {gray_path}"
    # Verify image properties
    rgb_img = fip.Image(str(rgb_path))
    assert rgb_img.channels == 3, "RGB image should have 3 channels"
    
    gray_img = fip.grayscale(fip.Image(str(gray_path)))
    assert gray_img.channels == 1, "Grayscale image should have 1 channel"

def test_grayscale_conversion(rgb_image):
    """Test grayscale conversion on different hardware."""
    result = fip.grayscale(rgb_image)
    
    # Check dimensions
    assert result.width == rgb_image.width
    assert result.height == rgb_image.height
    assert result.channels == 1
    
    # Save result for visual inspection if needed
    result.save(str(TEST_DIR / f"result_gray_cpu.png"))

@pytest.mark.parametrize("hardware", [
    Hardware.CPU,
    pytest.param(Hardware.GPU, marks=pytest.mark.skipif(
        not fip.is_gpu_available(), reason="GPU not available"))
])
def test_edge_detection(gray_image, hardware):
    """Test Sobel edge detection on different hardware."""
    # Test X gradient
    result_x = fip.edge_detect(gray_image, 1, 0, 3, hardware)
    assert result_x.channels == 1
    result_x.save(str(TEST_DIR / f"result_edge_x_{hardware.name.lower()}.png"))
    
    # Test Y gradient
    result_y = fip.edge_detect(gray_image, 0, 1, 3, hardware)
    assert result_y.channels == 1
    result_y.save(str(TEST_DIR / f"result_edge_y_{hardware.name.lower()}.png"))
    
    # Test combined XY gradient
    result_xy = fip.edge_detect(gray_image, 1, 1, 3, hardware)
    assert result_xy.channels == 1
    result_xy.save(str(TEST_DIR / f"result_edge_xy_{hardware.name.lower()}.png"))

def test_edge_detection_invalid_params(gray_image):
    """Test edge detection with invalid parameters."""
    with pytest.raises(ValueError):
        fip.edge_detect(gray_image, -1, 0, 3)
    
    with pytest.raises(ValueError):
        fip.edge_detect(gray_image, 0, 0, 3)
    
    with pytest.raises(ValueError):
        fip.edge_detect(gray_image, 1, 1, 2)