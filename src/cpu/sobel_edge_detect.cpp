#include "cpu/sobel_edge_detect.h"
#include "cpu/grayscale.h"
#include <logging/logging.h>
#include <algorithm>
#include <cmath>
#include <execution>
#include <vector>
#include <array>
#include <iostream>

namespace {
    // Preset Sobel kernels
    // Derivative kernels (dx=1 or dy=1) for calculating gradient
    constexpr std::array<float, 3> sobel_3 = {-1, 0, 1};
    constexpr std::array<float, 5> sobel_5 = {-1, -2, 0, 2, 1};
    constexpr std::array<float, 7> sobel_7 = {-1, -4, -5, 0, 5, 4, 1};

    // Smoothing kernels (dx=0 or dy=0) for reducing noise
     // Normalized by sum = 4
    constexpr std::array<float, 3> smooth_3 = {1.f/4, 2.f/4, 1.f/4};

    // Normalized by sum = 16
    constexpr std::array<float, 5> smooth_5 = {1.f/16, 4.f/16, 6.f/16, 4.f/16, 
                                               1.f/16}; 

    // Normalized by sum = 50
    constexpr std::array<float, 7> smooth_7 = {1.f/50, 6.f/50, 15.f/50, 
                                               20.f/50, 15.f/50, 6.f/50, 1.f/50}; 

    // Get kernel based on order of derivative and kernel size
    const float* get_kernel(int derivative_order, int kernel_size) {
        if (derivative_order == 0) {
            switch(kernel_size) {
                case 3: return smooth_3.data();
                case 5: return smooth_5.data();
                case 7: return smooth_7.data();
                default: return nullptr;
            }
        } else {  // derivative_order == 1
            switch(kernel_size) {
                case 3: return sobel_3.data();
                case 5: return sobel_5.data();
                case 7: return sobel_7.data();
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

    // Sobel operator implemented using separable convolution
    void apply_sobel(const unsigned char* input, unsigned char* output,
                    int width, int height,
                    int dx, int dy, int kernel_size) {
        const float* kernel_deriv = get_kernel(1, kernel_size); // Derivative kernel
        const float* kernel_smooth = get_kernel(0, kernel_size);  // Smoothing kernel
        // Add kernel pointer validation
        if (!kernel_deriv || !kernel_smooth) {
            LOG(ERROR, "Failed to generate kernels. Possible invalid config");
            throw std::runtime_error(
                "Failed to generate kernels. Possible invalid config");
        }
        const int radius = kernel_size / 2;

        print_kernel(kernel_deriv, kernel_size, "Derivative kernel");
        print_kernel(kernel_smooth, kernel_size, "Smoothing kernel");

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

                for(int k = -radius; k <= radius; ++k) {
                    //X direction: input[y * width + px] is a pixel
                    const int px = std::clamp(x + k, 0, width - 1);
                    // For X gradient:
                    if (dx) {
                        // apply derivative kernel in X-direction if dx=1,
                        sum_x += input[y * width + px] * kernel_deriv[k + radius];
                    } else {
                        // apply smoothing kernel in X-direction if dx=0
                        sum_x += input[y * width + px] * kernel_smooth[k + radius];
                    }
                    // For Y gradient: always apply smoothing in Y-direction
                    sum_y += input[y * width + px] * kernel_smooth[k + radius];
                }
                temp[idx] = std::make_pair(sum_x, sum_y);
            });

        // Apply vertical convolution (Y-direction) and compute gradients
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [&temp, output, width, height, kernel_deriv, kernel_smooth, radius, dx, dy]
            (int idx) {
                const int x = idx % width;
                const int y = idx / width;
                float grad_x = 0.0f;
                float grad_y = 0.0f;

                for(int k = -radius; k <= radius; ++k) {
                    const int py = std::clamp(y + k, 0, height - 1); // Y-direction
                    // For X gradient:
                    // always apply smoothing in Y-direction 
                    grad_x += temp[py * width + x].first * kernel_smooth[k + radius];

                    // For Y gradient:
                    if (dy) {
                        // apply derivative kernel in Y-direction if dy=1,
                        grad_y += temp[py * width + x].second * kernel_deriv[k + radius];
                    } else {
                        // apply smoothing kernel if dy=0
                        grad_y += temp[py * width + x].second * kernel_smooth[k + radius];
                    }
                    
                }

                // Compute final gradient magnitude
                float magnitude;
                if (dx && dy) {
                    magnitude = std::sqrt(grad_x * grad_x + grad_y * grad_y);
                } else if (dx) {
                    magnitude = std::abs(grad_x);
                } else {
                    magnitude = std::abs(grad_y);
                }

                output[idx] = static_cast<unsigned char>(
                    std::clamp(magnitude, 0.0f, 255.0f));
            });
    }
} // local namespace

namespace cpu {
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