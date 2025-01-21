/**
 * @file sobel_edge_detect.h
 * @brief Interface to perform Sobel Edge Detection CPU.
*/
#ifndef SOBEL_EDGE_DETECT_H
#define SOBEL_EDGE_DETECT_H
#include "image/image.h"

namespace cpu {

    /**
     * Performs Sobel Edge Detection with configurable direction and kernel size on the CPU.
     *
     * @param input The input image of type Image::Image (must be 1 or 3 channels).
     * @param dx: order of derivative in x direction (0 or 1 supported).
     *            performs edge detection on X-Axis if set to 1, else performs smoothing
     * @param dy: order of derivative in Y direction (0 or 1 supported).
     *            performs edge detection on Y-Axis if set to 1, else performs smoothing
     * @param kernel size: An integer that specifies a square kernel (must be odd and >= 1 and <= 7)
     *                     Default size is 3x3
     * @return Resultant mask image after applying Sobel filters.
    *  @throws std::invalid_argument if invalid kernel size is passed (supported = 1, 3, 5, and 7)
     */
    Image sobel_edge_detect(const Image& input, 
                          int dx = 1, 
                          int dy = 1,
                          int kernel_size = 3);
} // namespace cpu

#endif // SOBEL_EDGE_DETECT_H