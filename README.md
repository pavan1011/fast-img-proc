# Fast Image Processing Library

A memory-efficient image processing library implemented in C++, with GPU acceleration capabilities using CUDA. Python bindings provide a Pythonic interface to the library. Supported image processing operations are:

- Grayscale Conversion
- Histogram Equalization
- Edge Detection
- Blur

## Install

```bash
git clone --recurse-submodules git@github.com:pavan1011/fast-img-proc
cd fast-img-proc && mkdir build && cd build
```

### Configure without CUDA support (default)

```bash
cmake -S ../ -B .
```

```-DPYTHON_EXECUTABLE``` provides a hint to the CMake build system to help it find a specific version of Python.

```bash
cmake -S ../ -B . --DPYTHON_EXECUTABLE=<path/to/python>
```

### Configure with CUDA support
Requires CUDA compiler installed

```bash
cmake -S ../ -B . -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc
```
### Build Project
```bash
cmake --build .
```

## Usage in Python


```python
import fast_image_processing as fip

def main():
    # Load an RGB image (PNG or JPG supported)
    input_image = fip.Image("input.png")

    #Check if GPU with CUDA available
    print(f"GPU Available: {fip.is_gpu_available()}")
    print(f"Active Hardware: {fip.get_active_hardware()}")

    # Convert to grayscale using automatic hardware selection (default)
    auto_grayscale = fip.grayscale(input_image)

    # Save resultant image
    auto_grayscale.save("grayscale_auto.png")

    # Convert to grayscale using CPU
    cpu_grayscale = fip.grayscale(input_image, fip.Hardware.CPU)
    cpu_grayscale.save("grayscale_cpu.png")

    # Convert to grayscale using GPU
    try:
        gpu_grayscale = fip.grayscale(input_image, fip.Hardware.GPU)
        gpu_grayscale.save("grayscale_gpu.png")
    except RuntimeError as ex:
        printf(f"GPU processing failed: {ex}")
    
    # Equalize Histogram for RGB image using CPU
    cpu_hist_equalized_rgb = fip.equalize_histogram(input_image, fip.Hardware.CPU)
    cpu_hist_equalized_rgb.save("hist_equalized_rgb_cpu.png")

    # Equalize Histogram for Grayscale image using CPU
    cpu_hist_equalized_gray = fip.equalize_histogram(cpu_grayscale, fip.Hardware.CPU)
    cpu_hist_equalized_gray.save("hist_equalized_gray_cpu.png")

    # Blur image using CPU
    cpu_blur = fip.blur(input_image, 5, 1.4, fip.Hardware.CPU)
    cpu_blur.save("blur_cpu.png")
```



