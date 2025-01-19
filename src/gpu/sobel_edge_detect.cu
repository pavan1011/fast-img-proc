#include "gpu/sobel_edge_detect.cuh"
#include "gpu/grayscale.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <array>
#include <iostream>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG(ERROR, "CUDA error: {}", cudaGetErrorString(err)); \
        throw std::runtime_error(error_msg); \
    } \
} while(0)

namespace {
    // Preset Sobel kernels
    // Derivative kernels (dx=1 or dy=1) for calculating gradient
    constexpr std::array<float, 3> sobel_3 = {-1, 0, 1};
    constexpr std::array<float, 5> sobel_5 = {-1, -2, 0, 2, 1};
    constexpr std::array<float, 7> sobel_7 = {-1, -4, -5, 0, 5, 4, 1};

    // Smoothing kernels (dx=0 or dy=0) for reducing noise
    constexpr std::array<float, 3> smooth_3 = {1.f/4, 2.f/4, 1.f/4};
    constexpr std::array<float, 5> smooth_5 = {1.f/16, 4.f/16, 6.f/16, 4.f/16, 1.f/16};
    constexpr std::array<float, 7> smooth_7 = {1.f/50, 6.f/50, 15.f/50, 20.f/50, 
                                              15.f/50, 6.f/50, 1.f/50};

    // Separate constant memory arrays for each kernel size
    __constant__ float d_kernel_deriv_3[3];
    __constant__ float d_kernel_deriv_5[5];
    __constant__ float d_kernel_deriv_7[7];
    __constant__ float d_kernel_smooth_3[3];
    __constant__ float d_kernel_smooth_5[5];
    __constant__ float d_kernel_smooth_7[7];

    // Get kernel based on order of derivative and kernel size
    const float* get_kernel(int derivative_order, int kernel_size) {
        if (derivative_order == 0) {
            switch(kernel_size) {
                case 3: return smooth_3.data();
                case 5: return smooth_5.data();
                case 7: return smooth_7.data();
                default: return nullptr;
            }
        } else {
            switch(kernel_size) {
                case 3: return sobel_3.data();
                case 5: return sobel_5.data();
                case 7: return sobel_7.data();
                default: return nullptr;
            }
        }
    }

    // Helper function to get appropriate device kernel pointer
    __device__ const float* get_device_kernel(bool is_derivative, int kernel_size) {
        if (is_derivative) {
            switch(kernel_size) {
                case 3: return d_kernel_deriv_3;
                case 5: return d_kernel_deriv_5;
                case 7: return d_kernel_deriv_7;
                default: return nullptr;
            }
        } else {
            switch(kernel_size) {
                case 3: return d_kernel_smooth_3;
                case 5: return d_kernel_smooth_5;
                case 7: return d_kernel_smooth_7;
                default: return nullptr;
            }
        }
    }

    __global__ void sobel_kernel(cudaTextureObject_t tex_input,
                                unsigned char* output, int width, int height,
                                int dx, int dy, int kernel_size) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        const int radius = kernel_size / 2;
        const float* kernel_dx = get_device_kernel(true, kernel_size);
        const float* kernel_smooth = get_device_kernel(false, kernel_size);
        
        float sum_x = 0.0f;
        float sum_y = 0.0f;

        // Horizontal pass
        for(int k = -radius; k <= radius; ++k) {
            float pixel = tex2D<unsigned char>(tex_input, x + k, y);
            if(dx) {
                sum_x += pixel * kernel_dx[k + radius];
            } else {
                sum_x += pixel * kernel_smooth[k + radius];
            }
            sum_y += pixel * kernel_smooth[k + radius];
        }

        float grad_x = 0.0f;
        float grad_y = 0.0f;

        // Vertical pass
        for(int k = -radius; k <= radius; ++k) {
            float temp_x = tex2D<unsigned char>(tex_input, x, y + k);
            grad_x += sum_x * kernel_smooth[k + radius];

            if(dy) {
                grad_y += sum_y * kernel_dx[k + radius];
            } else {
                grad_y += sum_y * kernel_smooth[k + radius];
            }
        }

        // Compute magnitude
        float magnitude;
        if(dx && dy) {
            magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y);
        } else if(dx) {
            magnitude = fabsf(grad_x);
        } else {
            magnitude = fabsf(grad_y);
        }

        output[y * width + x] = static_cast<unsigned char>(
            min(255.0f, max(0.0f, magnitude)));
    }

    // Helper function to copy appropriate kernel to constant memory
    void copy_kernel_to_device(const float* kernel, int kernel_size, bool is_derivative) {
        if (is_derivative) {
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
        } else {
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
        }
    }
}

namespace gpu {
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
        Image gray_image(width, height, 1);  // Will only be used if needed

        if (input.channels() == 1) {
            source_data = input.data();
        } else {
            gray_image = gpu::grayscale(input);  // Creates new Image
            source_data = gray_image.data();
        }

        // Get kernels
        const float* kernel_dx = get_kernel(1, kernel_size);
        const float* kernel_smooth = get_kernel(0, kernel_size);

        // Copy kernels to appropriate constant memory arrays
        copy_kernel_to_device(kernel_dx, kernel_size, true);
        copy_kernel_to_device(kernel_smooth, kernel_size, false);

        // Initialize device resources
        cudaTextureObject_t tex_input = 0;
        cudaArray* d_array = nullptr;
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

            // Allocate output buffer
            CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(unsigned char)));

            // Launch kernel
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x,
                     (height + block.y - 1) / block.y);

            sobel_kernel<<<grid, block>>>(tex_input, d_output, width, height, 
                                        dx, dy, kernel_size);
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
            if (tex_input) cudaDestroyTextureObject(tex_input);
            if (d_array) cudaFreeArray(d_array);
            if (d_output) cudaFree(d_output);
            LOG(ERROR , "Error in GPU Sobel edge detect!");
            throw;
        }
    }
} // namespace gpu