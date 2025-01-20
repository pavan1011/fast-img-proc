#include "gpu/grayscale.cuh"
#include "logging/logging.h"
#include <cuda_runtime.h>
#include <iostream>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::string error_msg = std::string(__FILE__) + \
                               " line " + std::to_string(__LINE__) + ": " + \
                               cudaGetErrorString(err); \
        LOG(ERROR, "CUDA error in: {}", error_msg); \
        throw std::runtime_error(error_msg); \
    } \
} while(0)
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
    static constexpr int BLOCK_SIZE  = 256;

    bool is_available() {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        return deviceCount > 0;
    }

    Image grayscale(const Image& input) {
        LOG(DEBUG, "GPU: Starting Grayscale conversion.");
        if (!is_available()) {
            // TODO: Remove throw and return error code instead.
            LOG(ERROR, "CUDA device not available");
            throw std::runtime_error("CUDA device not available");
        }

        const auto width = input.width();
        const auto height = input.height();
        const auto channels = input.channels();
        const auto size = width * height;
        // cudaError_t error;

        if (channels != 3) {
                // TODO: Remove throw and return error code instead.
                LOG(ERROR, "Input image must have 3 channels");
                throw std::invalid_argument("Input image must have 3 channels");
        }
        Image output(width, height, 1);
        unsigned char *d_input, *d_output;

        try{

            CUDA_CHECK(cudaMalloc(&d_input, size * channels * sizeof(unsigned char)));
            CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(unsigned char)));

            CUDA_CHECK(cudaMemcpy(d_input, input.data(),
                    size * channels * sizeof(unsigned char),
                    cudaMemcpyHostToDevice));

            const int numBlocks = (size + BLOCK_SIZE  - 1) / BLOCK_SIZE ;

            grayscale_kernel<<<numBlocks, BLOCK_SIZE >>>(
                d_input, d_output, width, height, channels);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(output.data(), d_output,
                            size * sizeof(unsigned char),
                            cudaMemcpyDeviceToHost));

            LOG(DEBUG, "GPU: Grayscale conversion done.");
            return output;

        }catch(...){
            // Prevent memory leaks in case any unhandled exception occurs
            if(d_input) cudaFree(d_input);
            if(d_output) cudaFree(d_output);
            LOG(ERROR , "Error in GPU grayscale!");
            throw;
        }
        
    }
} // namespace gpu