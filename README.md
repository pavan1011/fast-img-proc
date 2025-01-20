# ```fast-img-proc```: A fast image processing library

A fast and memory-efficient image processing library utilizing parallel programming on CPU implemented in C++, with GPU acceleration capabilities using CUDA. Bindings generated using ```nanobind``` provide a Pythonic interface to the library. 

The supported image processing operations are:

- Grayscale Conversion
- Histogram Equalization
- Edge Detection
- Blur

The supported image formats are:
- PNG
- JPG


## Required Software Packages
- **C++ (required)**: compiler that supports C++20
- **CMake 3.18+ (required)** : ```sudo apt-get -y install cmake```
- **Python 3.7+ (required)**  : ```sudo apt install python3.7-dev``` [[github link](https://github.com/python/cpython)]
- **TBB (required)** : ```sudo apt-get install libtbb-dev``` [[github link](https://github.com/ibaned/tbb)]
- **nanobind (required)** : included as a git submodule in ```src/external/nanobind``` [[github link](https://github.com/wjakob/nanobind)]
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
cmake -DCMAKE_BUILD_TYPE=Release -S ../ -B .
```

#### CMake Build Flags

- ```-DCMAKE_BUILD_TYPE```: provides option to set the following types of builds: 

    - Debug: shows debug, info, warn, and, error logs
    - Profile: profile logs only
    - Verbose: info, warn, error logs
    - Release: warn and error logs only

- ```-DUSE_CUDA```: optionally enables GPU acceleration for supported image processing algorithms

- ```-DBUILD_DOCUMENTATION```: optionally enables detailed documentation generation locally using Doxygen 


- ```-DCMAKE_CUDA_COMPILER```: path to CUDA compiler. Required if ```-DUSE_CUDA``` is set to ON. Usually at ```/usr/local/cuda-<version>/bin/nvcc```

- ```-DCUDA_TOOLKIT_ROOT_DIR```: path to CUDA toolkit. Required if ```-DUSE_CUDA``` is set to ON. Usually at ```/usr/local/cuda-<version>```

- ```-DPYTHON_EXECUTABLE``` provides a hint to the CMake build system to help it find a specific version of Python (for virtual environments and non-default python installations).


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

    try:
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
    
    except RuntimeError as ex:
        printf(f"GPU processing failed: {ex}")


```
## Testing

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE=<path-to-python> -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc -S ../ -B .
```

```bash
cmake --build . --target cpp_tests
```


## Detailed Documentation

A detailed version of documentation of the source files, including class and member definitions, function signatures, and other implementation details can be generated locally using this project's source files.

### Install Doxygen and graphviz

```bash
sudo apt-get install doxygen graphviz
```

### Configure build to generate docs

```bash
cd fast-img-proc && mkdir build_docs && cd build_docs
cmake -S ../ -B . -DBUILD_DOCUMENTATION=ON 
```

### Build Documentation
```bash
cmake --build . --target docs
```

This will generate detailed documentation which can be viewed by opening ```fast-img-proc/build_docs/docs/html/html.index```.

## Credits

### Image Loading and Saving

The ```stb``` library from https://github.com/nothings/stb (MIT and Public Domain licenses) was used to populate ```fast-img-proc/external/stb```.

- `stb_image.h`: used to load images and represent them as buffers for further processing.
- `stb_image_write.h`: used to save images after processing.

### Python Bindings

The ```nanobind``` library from https://github.com/wjakob/nanobind (BSD-3-Clause license) was used to generate pythonic bindings to ```fast-img-proc``` C++ library.

### Detailed Documentation

The detailed documentation is generated locally using ```Doyxgen```: https://www.doxygen.nl/index.html.

The ```graphviz``` library from https://github.com/graphp/graphviz (MIT license) was used to generate dependency diagrams from this project's source files.

The ```doxygen-awesome-css``` library from https://github.com/jothepro/doxygen-awesome-css (MIT license) was used for custom styling in this documentation, namely: 

- ```fast-img-proc/docs/doxygen-awesome-sidebar-only.css```
- ```fast-img-proc/docs/doxygen-awesome.css```

**My special thanks to the authors and contributors of all the above libraries.**


