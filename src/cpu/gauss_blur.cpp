/**
 * @file gauss_blur.cpp
 * @brief CPU implementation of Gaussian blur using separable convolution
 */
#include "cpu/gauss_blur.h"
#include "logging/logging.h"
#include <vector>
#include <execution>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <numeric>
namespace {
    /** @brief Minimum supported kernel size */
    constexpr int MIN_KERNEL_SIZE = 3;

    /** @brief Maximum supported kernel size */
    constexpr int MAX_KERNEL_SIZE = UINT32_MAX >> 2;

    /**
     * @brief Creates 1D Gaussian kernel for separable convolution
     * 
     * Implements the Gaussian function:
     * G(x) = exp(-(x^2)/(2*sigma^2))
     * 
     * @param size Kernel size (must be odd)
     * @param sigma Standard deviation of Gaussian distribution
     * @return Normalized 1D Gaussian kernel
     * 
     * @note Kernel values are logged at DEBUG level
     */
    std::vector<float> create_gaussian_kernel(int size, float sigma) {
        std::vector<float> kernel(size);
        const int center = size / 2;
        float sum = 0.0f;

        // Calculate 1D kernel values for separable convolution
        for(int x = -center; x <= center; ++x) {
            float value = std::exp(-(x*x)/(2*sigma*sigma));
            kernel[x + center] = value;
            sum += value;
        }

        // Normalize kernel
        for(int i = 0; i < size; ++i) {
            kernel[i] /= sum;
        }

        // Print kernel values
        LOG_NNL(DEBUG, "Gaussian kernel (size= {}, sigma= {}):\n[", size, sigma);
        for(int i = 0; i < size; ++i) {
            LOG_NNL(DEBUG, "{}", kernel[i]);
            if(i < size-1) LOG_NNL(DEBUG, ", ");
        }
        LOG_NNL(DEBUG, "]\n");

        return kernel;
    }

    /**
     * @brief Applies separable Gaussian blur to a single channel
     * 
     * Implementation:
     * 1. Applies horizontal 1D convolution
     * 2. Stores results in temporary buffer
     * 3. Applies vertical 1D convolution
     * 4. Clamps results to [0,255]
     * 
     * @param input Source image data
     * @param output Destination image data
     * @param width Image width
     * @param height Image height
     * @param channels Number of color channels
     * @param channel_offset Current channel being processed
     * @param kernel 1D Gaussian kernel
     * 
     * @note Uses parallel execution via std::execution::par_unseq
     */
    void apply_gaussian_blur(const unsigned char* input, unsigned char* output,
                            int width, int height, int channels, int channel_offset,
                            const std::vector<float>& kernel) {
        const int kernel_size = kernel.size();
        const int radius = kernel_size / 2;

        std::vector<int> indices(width * height);
        std::iota(indices.begin(), indices.end(), 0);

        // Temporary buffer for separable convolution
        std::vector<float> temp(width * height);

        // Apply horizontal kernel
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [input, &temp, width, height, channels, channel_offset, &kernel, radius]
            (int idx) {
                const int x = idx % width;
                const int y = idx / width;
                float sum = 0.0f;

                // Apply horizontal kernel
                for(int k = -radius; k <= radius; ++k) {
                    const int px = std::clamp(x + k, 0, width - 1);
                    // use channel offset to choose the right channel
                    // being processed.
                    sum += input[(y * width + px) * channels + 
                        channel_offset] * kernel[k + radius];
                }
                temp[idx] = sum;
            });

        // Apply vertical kernel
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [&temp, output, width, height, channels, channel_offset, &kernel, radius]
            (int idx) {
                const int x = idx % width;
                const int y = idx / width;
                float sum = 0.0f;

                // Apply vertical kernel
                for(int k = -radius; k <= radius; ++k) {
                    const int py = std::clamp(y + k, 0, height - 1);
                    // Only need to account for channel offset when
                    // writing to output image
                    sum += temp[py * width + x] * kernel[k + radius];
                }
                output[idx * channels + channel_offset] =
                    static_cast<unsigned char>(std::clamp(sum + 0.5f, 0.0f, 255.0f));

            });
    }

} //local namespace

namespace cpu {
    /**
     * @brief Performs Gaussian blur on an image
     * 
     * Uses separable convolution for efficiency:
     * - 2D Gaussian filter is separated into two 1D convolutions
     * - Each channel is processed independently
     * - Border pixels are handled using clamp-to-edge strategy
     * 
     * Default sigma calculation (if sigma <= 0):
     * sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 1.0
     * 
     * @param input Source image
     * @param kernel_size Size of the Gaussian kernel (must be odd)
     * @param sigma Standard deviation of Gaussian distribution
     * @return Blurred image with same dimensions and channels
     * 
     * @throws std::invalid_argument if:
     * - kernel_size is even
     * - kernel_size < 3 or > MAX_KERNEL_SIZE
     * - image dimensions < kernel_size
     * 
     * @note Processes each channel sequentially but pixels in parallel
     */
    Image gaussian_blur(const Image& input, int kernel_size, float sigma) {
        LOG(DEBUG, "CPU: Starting Gaussian blur with ksize: {}, sigma: {}", 
                    kernel_size, sigma);
        // Validate kernel size
        if (kernel_size % 2 == 0) {
            LOG(ERROR, "Kernel size must be odd");
            throw std::invalid_argument("Kernel size must be odd.");
        }
        if (kernel_size < MIN_KERNEL_SIZE || kernel_size > MAX_KERNEL_SIZE) {
            // TODO: Test larger kernel size and enable if possible
            LOG(ERROR, "Kernel size supported: 3x3 up to {}x{}", MAX_KERNEL_SIZE, MAX_KERNEL_SIZE);
            throw std::invalid_argument("Kernel size supported: 3x3 up to max image width x height");
        }

        // Set default sigma based on kernel size if not specified.
        // From OpenCV implementation.
        if (sigma <= 0.0f) {
            sigma = 0.3f * ((kernel_size - 1) * 0.5f - 1) + 1.0f;
            LOG(WARN, "Blur: sigma value set to non-positive,"
                       "calculating default. Sigma: {}", sigma);
        }

        const auto width = input.width();
        const auto height = input.height();
        const auto channels = input.channels();

        // Validate image dimensions
        if (width < kernel_size || height < kernel_size) {
            LOG(ERROR, "Image dimensions must be at least kernel_size x kernel_size");
            throw std::invalid_argument("Image dimensions must be at "
                                        "least kernel_size x kernel_size");
        }

        auto kernel = create_gaussian_kernel(kernel_size, sigma);
        Image output(width, height, channels);

        // Process each channel sequentially
        for (int c = 0; c < channels; ++c) {
            apply_gaussian_blur(input.data(), output.data(),
                              width, height, channels, c, kernel);
        }

        LOG(DEBUG, "CPU: Gaussian blur done.");
        return output;
    }

} // namespace cpu