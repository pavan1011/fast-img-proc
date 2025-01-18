#include "gpu/grayscale.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace {
    __global__ void grayscale_kernel(const unsigned char* input, 
                                   unsigned char* output,
                                   int width, int height, int channels) {
        // Calculate global thread index
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

    Image grayscale(const Image& input) {
        if (!is_available()) {
            // TODO: Remove throw and return error code instead.
            throw std::runtime_error("CUDA device not available");
        }

        const auto width = input.width();
        const auto height = input.height();
        const auto channels = input.channels();
        const auto size = width * height;
        cudaError_t error;

        if (channels != 3) {
            // TODO: Remove throw and return error code instead.
            throw std::invalid_argument("Input image must have 3 channels");
        }

        Image output(width, height, 1);
        unsigned char *d_input, *d_output;
        cudaMalloc(&d_input, size * channels * sizeof(unsigned char));
        cudaMalloc(&d_output, size * sizeof(unsigned char));

        error = cudaMemcpy(d_input, input.data(),
                  size * channels * sizeof(unsigned char),
                  cudaMemcpyHostToDevice);

        // Check for errors, free d_input and d_output if error to prevent memleaks
        if (error != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            // TODO: Remove throw and return error code instead.
            throw std::runtime_error(std::string("Failed to copy data to GPU:") +
                                     cudaGetErrorString(error));
        }

        static constexpr int BLOCK_SIZE  = 256;
        const int numBlocks = (size + BLOCK_SIZE  - 1) / BLOCK_SIZE ;

        grayscale_kernel<<<numBlocks, BLOCK_SIZE >>>(
            d_input, d_output, width, height, channels);

        error = cudaGetLastError();
        // Check for errors, free d_input and d_output if error to prevent memleaks
        if (error != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            // TODO: Remove throw and return error code instead.
            throw std::runtime_error(std::string("Kernel exeuction failed:") +
                                     cudaGetErrorString(error));
        }

        error = cudaMemcpy(output.data(), d_output,
                          size * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost);

        // Check for errors, free d_input and d_output if error to prevent memleaks
        if (error != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            // TODO: Remove throw and return error code instead.
            throw std::runtime_error(std::string("Failed to copy data from GPU:") +
                                     cudaGetErrorString(error));
        }

        cudaFree(d_input);
        cudaFree(d_output);

        std::cout << "GPU: Grayscale conversion done.";

        return output;
    }
} // namespace gpu