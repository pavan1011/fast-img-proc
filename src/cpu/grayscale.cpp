#include "cpu/grayscale.h"
#include <execution>
#include <iostream>
#include <vector>

namespace cpu {

    Image grayscale(const Image& input) {
        // TODO: Remove throw and return error code instead. Pass output image by reference to grayscale().
        if (input.channels() == 1){
            throw std::runtime_error("Input image has only 1 channel. Already grayscale?");
        }else if(input.channels() != 3) {
            throw std::runtime_error("Image must have at least 3 channels");
        }
                
        Image output(input.width(), input.height(), 1);
        const auto size = input.width() * input.height();
        const auto in_data = input.data();
        const auto out_data = output.data();
        
        std::vector<int> indices(size);
        std::iota(indices.begin(), indices.end(), 0);
        
        // TODO: Create fallback option in case TBB is not installed
        // Makes calls to TBB library 
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [in_data, out_data, channels = input.channels()](int i) {
                const auto pixel = i * channels;
                out_data[i] = static_cast<unsigned char>(
                    0.299f * in_data[pixel] +
                    0.587f * in_data[pixel + 1] +
                    0.114f * in_data[pixel + 2]
                );
            });

        std::cout << "Grayscale conversion done!" << std::endl;  
        
        return output;
    }

} // namespace cpu