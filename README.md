# ```fast-img-proc```: A fast image processing library

A fast and memory-efficient image processing library utilizing parallel programming on CPU implemented in C++, with GPU acceleration capabilities using CUDA. Python bindings via ```nanobind``` provide a Pythonic interface to the library. Supported image processing operations are:

- Grayscale Conversion
- Histogram Equalization
- Edge Detection
- Blur

## Required Packages

- **FMT** (required): ```sudo apt install libfmt-dev``` [[github link](https://github.com/fmtlib/fmt)]
- **TBB** (required): ```sudo apt-get install libtbb-dev``` [[github link](https://github.com/ibaned/tbb)]
- **Python 3.7+** (required) : ```sudo apt install python3.7-dev``` [[github link](https://github.com/python/cpython)]
- **nanobind** (required): Included in ```src/external/nanobind``` [[github link](https://github.com/wjakob/nanobind)]
- **CUDA Toolkit & Driver** (optional):  [NVIDIA Installation guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)

## Installing ```fast-img-proc```

```bash
git clone --recurse-submodules git@github.com:pavan1011/fast-img-proc
cd fast-img-proc && mkdir build && cd build
```

### Configure default build  (without CUDA support)
```bash
cmake -S ../ -B .
```

**translates to**

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -S ../ -B .
```

#### CMake Build Flags

- ```-DCMAKE_BUILD_TYPE```: provides option to set the following types of builds: 

    - Debug: shows debug, info, warn, and, error logs
    - Profile: profile logs only
    - Verbose: info, warn, error logs
    - Release: warn and error logs only

- ```-DPYTHON_EXECUTABLE```: provides a hint to the CMake build system to help it find a specific version of Python. Useful if you want to point the build to a virtual environment or if Python is installed in a different directory instead of ```/usr/bin/python3```

- ```-DUSE_CUDA```: optionally enables GPU acceleration for supported image processing algorithms


- ```-DCMAKE_CUDA_COMPILER```: path to CUDA compiler. Required if ```-DUSE_CUDA``` is set to ON. Usually at ```/usr/local/cuda-<version>/bin/nvcc```

- ```-DCUDA_TOOLKIT_ROOT_DIR```: path to CUDA toolkit. Required if ```-DUSE_CUDA``` is set to ON. Usually at ```/usr/local/cuda-<version>```


### Configure with CUDA support
Requires CUDA compiler installed

```bash
cmake -S ../ -B . -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc
```
### Build ```fast-img-proc```
```bash
cmake --build .
```
Following the above steps generates ```fast_image_processing.cpython-<python-version>-<arch>-<platform>.so``` in ```/path/to/fast-img-proc/build```

#### Update PYTHONPATH

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/lib/directory
```


## Usage in Python

Detailed examples along with performance profiling shown in ```fast_img_proc/scripts```. 

Below is the basic usage:

```python
#  fast-img-proc exposed as fast_image_processing using nanobind
import fast_image_processing as fip

def main():
    # Load an RGB image (PNG or JPG supported)
    # using stbi_load from stb library
    input_image = fip.Image("input.png") 

    #Check if GPU with CUDA available
    print(f"GPU Available: {fip.is_gpu_available()}")
    print(f"Active Hardware: {fip.get_active_hardware()}")

    # Examples using automatic hardware selection

    # Convert to grayscale using automatic hardware selection (default)
    auto_grayscale = fip.grayscale(input_image)
    # Save resultant grayscale image
    auto_grayscale.save("grayscale_auto.png")

    # Apply histogram equalization using automatic hardware selection (default)
    auto_equalize_histogram = fip.equalize_histogram(input_image)
    auto_equalize_histogram.save("blur_auto.png")

    # Apply Gaussian blur using automatic hardware selection (default)
    auto_blur = fip.blur(input_image)
    auto_blur.save("grayscale_auto.png")
    
    # Apply Sobel edge detection using automatic hardware selection (default)

    # Derivative on x-axis, smoothing on y-axis, kernel_size = 5x5
    auto_edge_detect_1_0_5 = edge_detect(input_image, 1, 0, 5, fip.Hardware.CPU)
    auto_edge_detect.save("auto_edge_detect_1_0_5.png")

    # Examples using CPU

    # Convert to grayscale on CPU
    cpu_grayscale = fip.grayscale(input_image, fip.Hardware.CPU)
    cpu_grayscale.save("grayscale_cpu.png")

    # Equalize Histogram of an RGB image on CPU
    cpu_hist_equalized_rgb = fip.equalize_histogram(input_image, fip.Hardware.CPU)
    cpu_hist_equalized_rgb.save("hist_equalized_rgb_cpu.png")

    # Equalize Histogram of a grayscale image on CPU
    cpu_hist_equalized_gray = fip.equalize_histogram(cpu_grayscale, fip.Hardware.CPU)
    cpu_hist_equalized_gray.save("hist_equalized_gray_cpu.png")

    # Edge detection on CPU
    cpu_edge_1_0_5 = fip.edge_detect(input_image, 1, 0, 5, fip.Hardware.CPU)
    cpu_edge_1_0_5.save("edge_1_0_5_cpu.png")

    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_0_1_5 = fip.edge_detect(input_image, 0, 1, 5, fip.Hardware.CPU)
    cpu_edge_0_1_5.save("edge_0_1_5_cpu.png")

    # Derivatives on x-axis and y axis, kernel_size = 5
    cpu_edge_1_1_5 = fip.edge_detect(input_image, 0, 1, 5, fip.Hardware.CPU)
    cpu_edge_1_1_5.save("edge_0_1_5_cpu.png")

    # Convert to grayscale image using GPU
    try:
        gpu_grayscale = fip.grayscale(input_image, fip.Hardware.GPU)
        gpu_grayscale.save("grayscale_gpu.png")
    except RuntimeError as ex:
        printf(f"GPU processing failed: {ex}")

    # Edge detection on GPU
    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5x5
    gpu_edge_1_0_5 = fip.edge_detect(input_image, 1, 0, 5, fip.Hardware.GPU)
    gpu_edge_1_0_5.save("edge_1_0_5_gpu.png")

    # Smoothing on x-axis, derivative on y-axis, kernel_size = 5x5
    gpu_edge_0_1_5 = fip.edge_detect(input_image, 0, 1, 5, fip.Hardware.GPU)
    gpu_edge_0_1_5.save("edge_0_1_5_gpu.png")

    # Derivative on x and y axis, kernel_size = 5x5
    gpu_edge_1_1_5 = fip.edge_detect(input_image, 0, 1, 5, fip.Hardware.GPU)
    gpu_edge_1_1_5.save("edge_0_1_5_gpu.png")
```



