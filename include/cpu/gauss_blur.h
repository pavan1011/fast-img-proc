#ifndef GAUSS_BLUR_H
#define GAUSS_BLUR_H
#include "image/image.h"

namespace cpu {
    // Gaussian blur with configurable kernel size
    // kernel_size must be odd and >= 3
    // Default values correspond to 3x3 implementation
    Image gaussian_blur(const Image& input, 
                       int kernel_size = 3, 
                       float sigma = 1.0f);
}

#endif