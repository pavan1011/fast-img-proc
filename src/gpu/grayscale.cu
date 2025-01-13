#include "gpu/grayscale.cuh"
#include <cuda_runtime.h>

namespace {
    __global__ void grayscale_kernel(const unsigned char* input, 
                                   unsigned char* output,
                                   int width, int height, int channels) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= width * height) return;
        
        const int pixel = idx * channels;
        output[idx] = static_cast<unsigned char>(
            0.299f * input[pixel] +
            0.587f * input[pixel + 1] +
            0.114f * input[pixel + 2]
        );
    }
}

namespace gpu {
    bool is_available() {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        return deviceCount > 0;
    }

    Image to_grayscale(const Image& input) {
        if (!is_available()) {
            // TODO: Remove throw and return error code instead. Pass output image by reference to grayscale(). 
            throw std::runtime_error("CUDA device not available");
        }
        
        const auto size = input.width() * input.height();
        Image output(input.width(), input.height(), 1);
        
        unsigned char *d_input, *d_output;
        cudaMalloc(&d_input, size * input.channels() * sizeof(unsigned char));
        cudaMalloc(&d_output, size * sizeof(unsigned char));
        
        cudaMemcpy(d_input, input.data(), 
                  size * input.channels() * sizeof(unsigned char), 
                  cudaMemcpyHostToDevice);
        
        const int blockSize = 256;
        const int numBlocks = (size + blockSize - 1) / blockSize;
        
        grayscale_kernel<<<numBlocks, blockSize>>>(
            d_input, d_output, input.width(), input.height(), input.channels());

        error = cudaGetLastError();
        // Check for errors, free d_input and d_output if error to prevent memleaks
        if (error != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            // TODO: Remove throw and return error code instead. Pass output image by reference to grayscale().
            throw std::runtime_error("Kernel execution failed");
        }
        
        error = cudaMemcpy(output.data(), d_output, 
                          size * sizeof(unsigned char), 
                          cudaMemcpyDeviceToHost);

        // Check for errors, free d_input and d_output if error to prevent memleaks
        if (error != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            // TODO: Remove throw and return error code instead. Pass output image by reference to grayscale().
            throw std::runtime_error("Failed to copy data from GPU");
        }
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        return output;
    }
}