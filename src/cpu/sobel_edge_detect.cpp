/**
 * @file sobel_edge_detect.cpp
 * @brief CPU implementation of Sobel edge detection using separable convolution
 */
#include "cpu/sobel_edge_detect.h"
#include "cpu/grayscale.h"
#include <logging/logging.h>
#include <algorithm>
#include <cmath>
#include <execution>
#include <vector>
#include <array>
#include <iostream>
#include <numeric>

namespace {
    // Normalization factor = SUM of smoothing kernel values.

    /** @brief Normalization factor for 3x3 kernel */
    constexpr uint8_t NORM_FACTOR_3 = 4;
    /** @brief Normalization factor for 5x5 kernel */
    constexpr uint8_t NORM_FACTOR_5 = 16;
    /** @brief Normalization factor for 7x7 kernel */
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
    
    /**
     * @brief Gets normalization factor based on kernel size
     * @param kernel_size Size of the kernel (3, 5, or 7)
     * @return Normalization factor for gradient magnitude
     */
    float get_norm_factor(int kernel_size) {
        switch(kernel_size) {
            case 3: return NORM_FACTOR_3;
            case 5: return NORM_FACTOR_5;
            case 7: return NORM_FACTOR_7;
            default: return 1.0f;
        }
    }

     /**
     * @brief Gets appropriate kernel based on derivative order and size
     * @param derivative_order Order of derivative (0 for smoothing, 1 for gradient)
     * @param kernel_size Size of kernel (3, 5, or 7)
     * @return Pointer to kernel data, nullptr if invalid parameters
     */
    const float* get_kernel(int derivative_order, int kernel_size) {
        if (derivative_order == 0) {
            switch(kernel_size) {
                case 3: return smooth_3.data();
                case 5: return smooth_5.data();
                case 7: return smooth_7.data();
                default: return nullptr;
            }
        } else {  // derivative_order == 1 is supported
            switch(kernel_size) {
                case 3: return sobel_3.data();
                case 5: return sobel_5.data();
                case 7: return sobel_7.data();
                default: return nullptr;
            }
        }
    }

    /**
     * @brief Logs kernel values for debugging
     * @param kernel Pointer to kernel data
     * @param size Size of kernel
     * @param name Name of kernel for logging
     */
    void print_kernel(const float* kernel, int size, const char* name) {
        LOG_NNL(DEBUG, "{}  (size {}): \n [",  name, size);
        for (int i = 0; i < size; ++i) {
            LOG_NNL(DEBUG, "{}",  kernel[i]);
            if (i < size-1) LOG_NNL(DEBUG, ", ");
        }
        LOG_NNL(DEBUG, "]\n");
    }

    /**
     * @brief Applies Sobel operator using separable convolution
     * 
     * Implementation steps:
     * 1. Horizontal pass:
     *    - Applies derivative kernel for dx=1
     *    - Applies smoothing kernel for dy=1
     * 2. Vertical pass:
     *    - Completes X-gradient with smoothing
     *    - Completes Y-gradient with derivative
     * 3. Computes gradient magnitude
     * 
     * @param input Source image data (grayscale)
     * @param output Destination for edge detection result
     * @param width Image width
     * @param height Image height
     * @param dx Order of X derivative (0 or 1)
     * @param dy Order of Y derivative (0 or 1)
     * @param kernel_size Size of Sobel kernel
     * 
     * @throws std::runtime_error if kernel generation fails
     * @note Uses parallel execution via std::execution::par_unseq
     */
    void apply_sobel(const unsigned char* input, unsigned char* output,
                    int width, int height, int dx, int dy, int kernel_size) {
        // Get appropriate kernels based on size
        const float* kernel_deriv = get_kernel(1, kernel_size); // Derivative kernel
        const float* kernel_smooth = get_kernel(0, kernel_size);  // Smoothing kernel
        
        // Add kernel pointer validation
        if (!kernel_deriv || !kernel_smooth) {
            LOG(ERROR, "Failed to generate kernels. Possible invalid config");
            throw std::runtime_error(
                "Failed to generate kernels. Possible invalid config");
        }
        
        print_kernel(kernel_deriv, kernel_size, "Derivative kernel");
        print_kernel(kernel_smooth, kernel_size, "Smoothing kernel");
        
        // radius = kernel_size/2
        const int radius = kernel_size >> 1;
        // Get normalization factor based on kernel size
        const float norm_factor = get_norm_factor(kernel_size);

        //Create indices to acces each pixel
        std::vector<int> indices(width * height);
        std::iota(indices.begin(), indices.end(), 0);

        // Temporary buffer for intermediate results
        // for X and Y axis gradients during separable convolution
        std::vector<std::pair<float, float>> temp(width * height);

        // Apply horizontal convolution (X-direction)
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [input, &temp, width, height, kernel_deriv, kernel_smooth, radius, dx, dy]
            (int idx) {
                const int x = idx % width;
                const int y = idx / width;
                float sum_x = 0.0f;
                float sum_y = 0.0f;

                // input[y * width + px] is a pixel
                for(int k = -radius; k <= radius; ++k) {
                    const int px = std::clamp(x + k, 0, width - 1); // X-direction
                    // Handle X-direction processing
                    if(dx) {
                        // When dx=1: Apply derivative in X-direction in horizontal pass
                        // This is the first part of Sobel X-gradient calculation
                        sum_x += input[y * width + px] * kernel_deriv[k + radius];
                    } else {
                        // When dx=0: Apply smoothing in X-direction
                        // This smoothing is needed for Y-gradient calculation
                        sum_x += input[y * width + px] * kernel_smooth[k + radius];
                    }

                    // Handle Y-direction processing
                    // For Y-direction, apply horizontal smoothing in the first pass
                    // This is the first step of the separable convolution for Y-gradient
                    sum_y += input[y * width + px] * kernel_smooth[k + radius];

                }
                temp[idx] = std::make_pair(sum_x, sum_y);
            });

        // Apply vertical convolution (Y-direction) and compute gradients
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [&temp, output, width, height, kernel_deriv, kernel_smooth, radius, dx, dy, norm_factor]
            (int idx) {
                const int x = idx % width;
                const int y = idx / width;
                float grad_x = 0.0f;
                float grad_y = 0.0f;

                for(int k = -radius; k <= radius; ++k) {
                    const int py = std::clamp(y + k, 0, height - 1); // Y-direction
                    // Complete X-gradient calculation
                    if(dx){
                        // When dx=1: Apply smoothing in Y-direction
                        // This completes the Sobel X-gradient calculation:
                        // (derivative in X) * (smoothing in Y)
                        // When dx=0: Do nothing since we already applied 
                        // smoothing in horizontal pass
                        grad_x += temp[py * width + x].first * kernel_smooth[k + radius];
                    }
                    // Complete Y-gradient calculation
                    if(dy) {
                        // When dy=1: Apply derivative in Y-direction
                        // This completes the Sobel Y-gradient calculation:
                        // (accumulated pixel values from horiz conv) * (derivative in Y)
                        grad_y += temp[py * width + x].second * kernel_deriv[k + radius];
                    } 
                    
                }

                // Compute final gradient magnitude
                float magnitude;
                if (dx && dy) {
                    magnitude = std::sqrt((grad_x * grad_x) + (grad_y * grad_y));
                } else if (dx) {
                    magnitude = std::abs(grad_x);
                } else {
                    magnitude = std::abs(grad_y);
                }

                // Apply normalization of gradients
                magnitude /= norm_factor;

                output[idx] = static_cast<unsigned char>(
                    std::clamp(magnitude, 0.0f, 255.0f));
            });
    }
} // local namespace

namespace cpu {
    /**
     * @brief Performs Sobel edge detection on an image
     * 
     * Supports:
     * - First-order derivatives (dx, dy = 0 or 1)
     * - Kernel sizes: 3x3, 5x5, 7x7 (1 defaults to 3x3)
     * - Automatic grayscale conversion for color images
     * 
     * @param input Source image
     * @param dx Order of X derivative (0 or 1)
     * @param dy Order of Y derivative (0 or 1)
     * @param kernel_size Size of Sobel kernel (1, 3, 5, or 7)
     * @return Single-channel image containing edge detection result
     * 
     * @throws std::invalid_argument if:
     * - dx or dy are not 0 or 1
     * - Both dx and dy are 0
     * - kernel_size is not 1, 3, 5, or 7
     * - Image dimensions < kernel_size
     * @note Uses kernel_size = 3 if kernel_size = 1 is passed 
     *       (similar to OpenCV implementation)
     */
    Image sobel_edge_detect(const Image& input, int dx, int dy, int kernel_size) {
        LOG(DEBUG, "CPU: Starting Sobel edge detection.");
        if (dx < 0 || dx > 1 || dy < 0 || dy > 1) {
            LOG(ERROR, "Invalid derivative order: dx and dy must be either 0 or 1");
            throw std::invalid_argument("dx and dy must be either 0 or 1");
        }
        if (dx == 0 && dy == 0) {
            LOG(ERROR, "Invalid derivative order: At least one of dx or dy must be 1");
            throw std::invalid_argument("At least one of dx or dy must be 1");
        }
        if (kernel_size % 2 == 0 || kernel_size < 1 || kernel_size > 7) {
            LOG(ERROR, "Kernel size must be 1, 3, 5, or 7");
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

        // Convert to grayscale if needed
        const unsigned char* source_data;
        Image gray_image(input.width(), input.height(), 1);

        if (input.channels() == 1) {
            source_data = input.data();
        } else {
            gray_image = cpu::grayscale(input);
            source_data = gray_image.data();
        }

        Image output(width, height, 1);
        apply_sobel(source_data, output.data(), width, height, 
                   dx, dy, kernel_size);
        
        LOG(DEBUG, "CPU Sobel edge detect done.");
        return output;
    }
} // namespace cpu