# Fast Image Processing Library

## Install

```bash
git clone https://github.com/pavan1011/fast-img-proc.git
cd fast-img-proc
```

### Configure without CUDA support (default)
```bash
cmake . -B build -DPython_EXECUTABLE=<path to python executable> ..
```

### Configure with CUDA support
Requires CUDA compiler installed
```bash
cmake . -B build -DPython_EXECUTABLE=<path to python executable> -DUSE_CUDA=<ON> ..
```

### Build Project
```bash
cmake --build build
```

## Usage in Python

### Grayscale Conversion

Convert an RGB image to grayscale. 

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
```



