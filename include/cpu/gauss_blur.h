/**
 * @file gauss_blur.h
 * @brief Interface to perform Gaussian Blur CPU.
*/
#ifndef GAUSS_BLUR_H
#define GAUSS_BLUR_H
#include "image/image.h"

namespace cpu {
    /**
     * Performs Gaussian blur with configurable kernel on the CPU.
     *
     * @param input The input image of type Image::Image.
     * @param kernel size: An integer that specifies a square kernel (must be odd and >= 3 and <= 29)
     *                     Default size is 3x3
     * @return Resultant image after applying gaussian blur.
    *  @throws std::invalid_argument if invalid kernel size is passed (supported = 1, 3, 5, and 7)
     */
    Image gaussian_blur(const Image& input, 
                       int kernel_size = 3, 
                       float sigma = 1.0f);

} // namespace cpu

#endif // HIST_EQUALIZE_H