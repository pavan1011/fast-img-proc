#include "cpu/gauss_blur.h"
#include <vector>
#include <execution>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>


namespace {
    std::vector<std::vector<float>> create_gaussian_kernel(int size, float sigma) {
        std::vector<std::vector<float>> kernel(size, std::vector<float>(size));
        const int center = size / 2;
        float sum = 0.0f;
        
        // Calculate kernel values
        for(int y = -center; y <= center; ++y) {
            for(int x = -center; x <= center; ++x) {
                float value = std::exp(-(x*x + y*y)/(2*sigma*sigma));
                kernel[y + center][x + center] = value;
                sum += value;
            }
        }
        
        // Normalize kernel
        for(int y = 0; y < size; ++y) {
            for(int x = 0; x < size; ++x) {
                kernel[y][x] /= sum;
            }
        }
        
        return kernel;
    }

    void apply_gaussian_blur(const unsigned char* input, unsigned char* output,
                            int width, int height, int channels, int channel_offset,
                            const std::vector<std::vector<float>>& kernel) {
        const int kernel_size = kernel.size();
        const int radius = kernel_size / 2;

        std::vector<int> indices(width * height);
        std::iota(indices.begin(), indices.end(), 0);

        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [input, output, width, height, channels, channel_offset, &kernel, radius]
            (int idx) {
                const int x = idx % width;
                const int y = idx / width;
                float sum = 0.0f;

                // Apply kernel
                for(int ky = -radius; ky <= radius; ++ky) {
                    for(int kx = -radius; kx <= radius; ++kx) {
                        const int px = std::clamp(x + kx, 0, width - 1);
                        const int py = std::clamp(y + ky, 0, height - 1);
                        const int pixel_idx = (py * width + px) * channels + channel_offset;
                        sum += input[pixel_idx] * kernel[ky + radius][kx + radius];
                    }
                }

                output[idx * channels + channel_offset] = 
                    static_cast<unsigned char>(sum + 0.5f);
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
            std::cout << "Blur: sigma value set to non-positive," 
                      << "calculating default." << std::endl;
            sigma = 0.3f * ((kernel_size - 1) * 0.5f - 1) + 1.0f;
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
} // cpu