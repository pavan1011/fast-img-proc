/**
 * @file grayscale.cpp
 * @brief CPU implementation of grayscale conversion
 */

#include "cpu/grayscale.h"
#include "logging/logging.h"
#include <execution>
#include <iostream>
#include <vector>
#include <numeric>
#include <cstring>

/**
 * @namespace cpu
 * @brief CPU implementations of image processing algorithms
 */
namespace cpu {
    /**
     * @brief Converts an RGB or grayscale image to grayscale using parallel processing
     * 
     * For RGB images, uses ITU-R BT.601 conversion formula:
     * gray = 0.299R + 0.587G + 0.114B
     * 
     * Implementation details:
     * - Uses parallel execution via std::execution::par_unseq
     * - Leverages TBB library for parallelization
     * - Performs data copy for single-channel images
     * 
     * @param input Source image (must be 1 or 3 channels)
     * @return Single-channel grayscale image
     * @throws std::runtime_error if input image has invalid number of channels
     * 
     * @note If input is already grayscale (1 channel), a copy is returned
     */
    Image grayscale(const Image& input) {
        LOG(DEBUG, "CPU: Starting Grayscale conversion.");
        // TODO: Remove throw and return error code instead.

        if (input.channels() != 1 && input.channels() != 3) {
            LOG(ERROR, "Input image must have 1 or 3 channels");
            throw std::runtime_error("Input image must have 1 or 3 channels");
        }
                
        Image output(input.width(), input.height(), 1);
        const auto size = input.width() * input.height();
        const auto in_data = input.data();
        const auto out_data = output.data();
        
        std::vector<int> indices(size);
        std::iota(indices.begin(), indices.end(), 0);

        if (input.channels() == 1){
            LOG(WARN, "Input image has only 1 channel. Is it already grayscale?");
            // For grayscale input, just copy the data
            input.processTiles(
                [](const unsigned char* in, unsigned char* out,
                   uint32_t width, uint32_t height, uint32_t stride, uint32_t) {
                    // Process each row of the tile
                    for (uint32_t y = 0; y < height; ++y) {
                        std::memcpy(out + y * stride, 
                                  in + y * stride, 
                                  width);
                    }
                }, output);
            return output;

        }else if(input.channels() != 3) {
            LOG(ERROR, "Input image must have 1 or 3 channels");
            throw std::runtime_error("Input image must have 1 or 3 channels");
        }
        
        // Convert pixels in parallel. STL makes calls to TBB library
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [in_data, out_data, channels = input.channels()](int i) {
                const auto pixel = i * channels;
                out_data[i] = static_cast<unsigned char>(
                    0.299f * in_data[pixel] +
                    0.587f * in_data[pixel + 1] +
                    0.114f * in_data[pixel + 2]
                );
            });

        LOG(DEBUG, "CPU: Grayscale conversion done.");
        
        return output;
    }

} // namespace cpu