#include "cpu/grayscale.h"
#include "logging/logging.h"
#include <execution>
#include <iostream>
#include <vector>
#include <numeric>

namespace cpu {

    Image grayscale(const Image& input) {
        LOG(DEBUG, "CPU: Starting Grayscale conversion.");
        // TODO: Remove throw and return error code instead.
                
        Image output(input.width(), input.height(), 1);
        const auto size = input.width() * input.height();
        const auto in_data = input.data();
        const auto out_data = output.data();
        
        std::vector<int> indices(size);
        std::iota(indices.begin(), indices.end(), 0);

        if (input.channels() == 1){
            LOG(WARN, "Input image has only 1 channel. Is it already grayscale?");
            std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                [in_data, out_data, channels = input.channels()](int i) {
                    out_data[i] = in_data[i];
                });
            return output;

        }else if(input.channels() != 3) {
            LOG(ERROR, "Input image must have at least 3 channels");
            throw std::runtime_error("Image must have at least 3 channels");
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