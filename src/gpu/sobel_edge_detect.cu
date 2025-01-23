#include "gpu/sobel_edge_detect.cuh"
#include "cpu/grayscale.h"
#include "logging/logging.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <array>
#include <iostream>


// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::string error_msg = std::string(__FILE__) + \
                               " line " + std::to_string(__LINE__) + ": " + \
                               cudaGetErrorString(err); \
        LOG(ERROR, "CUDA error: {}", error_msg); \
        throw std::runtime_error(error_msg); \
    } \
} while(0)

namespace {
    // CUDA-related constants
    // Preset CUDA Kernel block size (16 x 16)
    constexpr uint32_t KERNEL_BLK_DIM = 16;

    // Separate constant memory arrays for each kernel size
    __constant__ float d_kernel_deriv_3[3];
    __constant__ float d_kernel_deriv_5[5];
    __constant__ float d_kernel_deriv_7[7];
    __constant__ float d_kernel_smooth_3[3];
    __constant__ float d_kernel_smooth_5[5];
    __constant__ float d_kernel_smooth_7[7];

    // Normalization factor = SUM of smoothing kernel values.
    constexpr uint8_t NORM_FACTOR_3 = 4;
    constexpr uint8_t NORM_FACTOR_5 = 16;
    constexpr uint8_t NORM_FACTOR_7 = 64;

    // Preset Sobel kernels
    // Standard Sobel derivative kernels (dx=1 or dy=1) for calculating gradient
    constexpr std::array<float, 3> sobel_3 = {-1, 0, 1};
    constexpr std::array<float, 5> sobel_5 = {-1, -2, 0, 2, 1};
    constexpr std::array<float, 7> sobel_7 = {-1, -4, -5, 0, 5, 4, 1};

    // Smoothing kernels (dx=0 or dy=0) for reducing noise
    constexpr std::array<float, 3> smooth_3 = {1.f, 2.f, 1.f};
    constexpr std::array<float, 5> smooth_5 = {1.f, 4.f, 6.f, 4.f, 1.f};
    constexpr std::array<float, 7> smooth_7 = {1.f, 6.f, 15.f, 20.f, 15.f, 6.f, 1.f};

    // Get kernel based on order of derivative and kernel size
    const float* get_kernel(int derivative_order, int kernel_size) {
        if (derivative_order == 0) {
            switch(kernel_size) {
                case 3: return smooth_3.data();
                case 5: return smooth_5.data();
                case 7: return smooth_7.data();
                default: return nullptr;
            }
        } else { // derivative order == 1 is supported
            switch(kernel_size) {
                case 3: return sobel_3.data();
                case 5: return sobel_5.data();
                case 7: return sobel_7.data();
                default: return nullptr;
            }
        }
    }

    // Get normalization factor based on kernel size
    __device__ float get_norm_factor(int kernel_size) {
        switch(kernel_size) {
            case 3: return NORM_FACTOR_3;
            case 5: return NORM_FACTOR_5;
            case 7: return NORM_FACTOR_7;
            default: return 1.0f;
        }
    }

    // Helper function to get appropriate device kernel pointer
    __device__ const float* get_device_kernel(int derivative_order, int kernel_size) {
        if (derivative_order == 0) {
            switch(kernel_size) {
                case 3: return d_kernel_smooth_3;
                case 5: return d_kernel_smooth_5;
                case 7: return d_kernel_smooth_7;
                default: return nullptr;
            }
        } else {
            switch(kernel_size) {
                case 3: return d_kernel_deriv_3;
                case 5: return d_kernel_deriv_5;
                case 7: return d_kernel_deriv_7;
                default: return nullptr;
            }
        }
    }

    // Helper function to print chosen kernel
    void print_kernel(const float* kernel, int size, const char* name) {
        LOG_NNL(DEBUG, "{}  (size {}): \n [",  name, size);
        for (int i = 0; i < size; ++i) {
            LOG_NNL(DEBUG, "{}",  kernel[i]);
            if (i < size-1) LOG_NNL(DEBUG, ", ");
        }
        LOG_NNL(DEBUG, "]\n");
    }

    // Sobel kernels implemented using CUDA Texture and separable convolution
    // This kernel performs the horizontal pass
    __global__ void sobel_horizontal(cudaTextureObject_t tex_input,
                                   float* temp_sum_x, float* temp_sum_y,
                                   int width, int height, int kernel_size,
                                   int dx, int dy) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        const float* kernel_deriv = get_device_kernel(1, kernel_size);
        const float* kernel_smooth = get_device_kernel(0, kernel_size);
        const int radius = kernel_size >> 1;

        float sum_x = 0.0f;
        float sum_y = 0.0f;

        for(int k = -radius; k <= radius; ++k) {
            int px = min(width-1, max(x + k, 0));
            float pixel = tex2D<unsigned char>(tex_input, px, y);
            if (dx) {
                // For x derivative, use derivative kernel in x direction
                sum_x += pixel * kernel_deriv[k + radius];
            } else {
                sum_x += pixel * kernel_deriv[k + radius];
            }
            // For y derivative, use smoothing kernel in x direction
            sum_y += pixel * kernel_smooth[k + radius];
        }
        
        temp_sum_x[y * width + x] = sum_x;
        temp_sum_y[y * width + x] = sum_y;

    }

    // Sobel kernels implemented using CUDA Texture and separable convolution
    // This kernel performs the horizontal pass
    __global__ void sobel_vertical(float* temp_sum_x, float* temp_sum_y,
                                 unsigned char* output, int width, int height,
                                 int kernel_size, int dx, int dy) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        const int radius = kernel_size >> 1;
        const float norm_factor = get_norm_factor(kernel_size);
        const float* kernel_smooth = get_device_kernel(0, kernel_size);
        const float* kernel_deriv = get_device_kernel(1, kernel_size);
        
        float grad_x = 0.0f;
        float grad_y = 0.0f;

        for(int k = -radius; k <= radius; ++k) {
            int py = min(height-1, max(y + k, 0));
            if(py >= 0 && py < height) {
                if (dx) {
                    // Complete X gradient using smoothing in y direction
                    grad_x += temp_sum_x[py * width + x] * kernel_smooth[k + radius];
                }
                if (dy) {
                    // Complete Y gradient using derivative in y direction
                    grad_y += temp_sum_y[py * width + x] * kernel_deriv[k + radius];
                }
            }
        }

        float magnitude;
        if(dx && dy) {
            magnitude = sqrtf((grad_x * grad_x) + (grad_y * grad_y));
        } else if(dx) {
            magnitude = fabsf(grad_x);
        } else {
            magnitude = fabsf(grad_y);
        }

        magnitude /= norm_factor;

        output[y * width + x] = static_cast<unsigned char>(
            min(255.0f, max(0.0f, magnitude)));
    }

    // Helper function to copy appropriate kernel to constant memory
    void copy_kernel_to_device(const float* kernel, int kernel_size, int derivative_order) {
        if (derivative_order == 0) {
            switch(kernel_size) {
                case 3:
                    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_smooth_3, kernel, 
                                                3 * sizeof(float)));
                    break;
                case 5:
                    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_smooth_5, kernel, 
                                                5 * sizeof(float)));
                    break;
                case 7:
                    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_smooth_7, kernel, 
                                                7 * sizeof(float)));
                    break;
            }
        } else { // Currently, only derivative_order upto 1 supported
            switch(kernel_size) {
                case 3:
                    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_deriv_3, kernel, 
                                                3 * sizeof(float)));
                    break;
                case 5:
                    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_deriv_5, kernel, 
                                                5 * sizeof(float)));
                    break;
                case 7:
                    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_deriv_7, kernel, 
                                                7 * sizeof(float)));
                    break;
            }
        }
    }
}

/**
 * @namespace gpu
 * @brief GPU-accelerated image processing operations
 */
namespace gpu {
    // Helper function to check if CUDA device is available.
    // Expose to processing.cpp via gpu namespace
    bool is_available() {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        return deviceCount > 0;
    }
    
    Image sobel_edge_detect(const Image& input, int dx, int dy, int kernel_size) {
        LOG(DEBUG, "GPU: Starting Sobel edge detection.");
        if (!is_available()) {
            LOG(ERROR, "CUDA device not available");
            throw std::runtime_error("CUDA device not available");
        }

        if (dx < 0 || dx > 1 || dy < 0 || dy > 1) {
            LOG(ERROR, "Invalid derivative order: dx and dy must be either 0 or 1");
            throw std::invalid_argument("dx and dy must be either 0 or 1");
        }
        if (dx == 0 && dy == 0) {
            LOG(ERROR, "Invalid derivative order: At least one of dx or dy must be 1");
            throw std::invalid_argument("At least one of dx or dy must be 1");
        }
        if (kernel_size % 2 == 0 || kernel_size < 1 || kernel_size > 7) {
            LOG(ERROR, "Invalid kernel size. Must be 1, 3, 5, or 7");
            throw std::invalid_argument("Kernel size must be 1, 3, 5, or 7");
        }

        // Special case for kernel_size = 1,
        // OpenCV implements separable conv of 1x3 X 3x1 in this case.
        if (kernel_size == 1) kernel_size = 3;

        const auto width = input.width();
        const auto height = input.height();
        if (width < kernel_size || height < kernel_size) {
            LOG(ERROR, "Image dimensions must be at least kernel_size x kernel_size");
            throw std::invalid_argument(
                "Image dimensions must be at least kernel_size x kernel_size");
        }

        // Handle grayscale conversion
        const unsigned char* source_data;
        // Will only be used if grayscale conversion is needed
        Image gray_image(width, height, 1);  

        if (input.channels() == 1) {
            source_data = input.data();
        } else {
            gray_image = cpu::grayscale(input);  // Creates new Image
            source_data = gray_image.data();
        }

        // Get kernels
        const float* kernel_deriv = get_kernel(1, kernel_size);
        const float* kernel_smooth = get_kernel(0, kernel_size);

        print_kernel(kernel_deriv, kernel_size, "Derivative kernel");
        print_kernel(kernel_smooth, kernel_size, "Smoothing kernel");

        // Add kernel pointer validation
        if (!kernel_deriv || !kernel_smooth) {
            LOG(ERROR, "Failed to generate kernels. Possible invalid config");
            throw std::runtime_error(
                "Failed to generate kernels. Possible invalid config");
        }

        // Copy kernels to appropriate constant memory arrays
        copy_kernel_to_device(kernel_deriv, kernel_size, 1);
        copy_kernel_to_device(kernel_smooth, kernel_size, 0);

        // Initialize device resources
        cudaTextureObject_t tex_input = 0;
        cudaArray* d_array = nullptr;
        float* d_temp_sum_x = nullptr;
        float* d_temp_sum_y = nullptr;
        unsigned char* d_output = nullptr;

        try {
            // Create and copy to CUDA array
            cudaChannelFormatDesc channelDesc = 
                cudaCreateChannelDesc<unsigned char>();
            CUDA_CHECK(cudaMallocArray(&d_array, &channelDesc, width, height));
            
            CUDA_CHECK(cudaMemcpy2DToArray(d_array, 
                                         0, 0,
                                         source_data,
                                         width * sizeof(unsigned char),
                                         width * sizeof(unsigned char),
                                         height,
                                         cudaMemcpyHostToDevice));

            // Specify texture object parameters
            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = d_array;

            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = false;

            // Create texture object
            CUDA_CHECK(cudaCreateTextureObject(&tex_input, &resDesc, &texDesc, nullptr));

            // Allocate temporary buffer to store intermediate horizontal pass outputs
            CUDA_CHECK(cudaMalloc(&d_temp_sum_x, width * height * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_temp_sum_y, width * height * sizeof(float)));


            // Allocate output buffer
            CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(unsigned char)));

            // Launch kernel
            dim3 block(KERNEL_BLK_DIM, KERNEL_BLK_DIM);
            dim3 grid((width + block.x - 1) / block.x,
                     (height + block.y - 1) / block.y);

            LOG(DEBUG, "Calling sobel_kernel with width: {}, height: {},"
                       " dx: {}, dy: {}, kernel_size: {}", width, height, 
                                        dx, dy, kernel_size);

            sobel_horizontal<<<grid, block>>>(tex_input, d_temp_sum_x, 
                                            d_temp_sum_y, width, height, 
                                            kernel_size, dx, dy);
            CUDA_CHECK(cudaGetLastError());

            sobel_vertical<<<grid, block>>>(d_temp_sum_x, d_temp_sum_y, d_output, 
                                            width, height, kernel_size, dx, dy);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Create output image and copy result
            Image output(width, height, 1);
            CUDA_CHECK(cudaMemcpy(output.data(), d_output,
                                width * height * sizeof(unsigned char),
                                cudaMemcpyDeviceToHost));

            // Cleanup
            CUDA_CHECK(cudaDestroyTextureObject(tex_input));
            CUDA_CHECK(cudaFreeArray(d_array));
            CUDA_CHECK(cudaFree(d_output));

            LOG(DEBUG , "GPU: Sobel edge detect done.");
            return output;

        } catch (...) {
            // Prevent memory leaks in case any unhandled exception occurs
            if (tex_input) cudaDestroyTextureObject(tex_input);
            if (d_array) cudaFreeArray(d_array);
            if (d_output) cudaFree(d_output);
            LOG(ERROR , "Error in GPU Sobel edge detect!");
            throw;
        }
    }
} // namespace gpu