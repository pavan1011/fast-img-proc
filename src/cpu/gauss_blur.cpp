#include "cpu/gauss_blur.h"
#include <vector>
#include <execution>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>


namespace {
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
        std::cout << "Gaussian kernel (size=" << size << ", sigma=" << sigma << "):\n[";
        for(int i = 0; i < size; ++i) {
            std::cout << kernel[i];
            if(i < size-1) std::cout << ", ";
        }
        std::cout << "]\n";

        return kernel;
    }

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
    Image gaussian_blur(const Image& input, int kernel_size, float sigma) {
        // Validate kernel size
        if (kernel_size % 2 == 0) {
            throw std::invalid_argument("Kernel size must be odd.");
        }
        if (kernel_size < 3 || kernel_size > 30) {
            throw std::invalid_argument("Kernel size supported: 3x3 up to 29x29");
        }

        // Set default sigma based on kernel size if not specified.
        // From OpenCV implementation.
        if (sigma <= 0.0f) {
            sigma = 0.3f * ((kernel_size - 1) * 0.5f - 1) + 1.0f;
            std::cout << "Blur: sigma value set to non-positive,"
                      << "calculating default. Sigma:" << sigma << std::endl;

        }

        const auto width = input.width();
        const auto height = input.height();
        const auto channels = input.channels();

        // Validate image dimensions
        if (width < kernel_size || height < kernel_size) {
            throw std::invalid_argument("Image dimensions must be at least kernel_size x kernel_size");
        }

        auto kernel = create_gaussian_kernel(kernel_size, sigma);
        Image output(width, height, channels);

        // Process each channel sequentially
        for (int c = 0; c < channels; ++c) {
            apply_gaussian_blur(input.data(), output.data(),
                              width, height, channels, c, kernel);
        }

        return output;
    }

} // namespace cpu