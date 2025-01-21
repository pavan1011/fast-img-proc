/**
 * @file grayscale.h
 * @brief Interface to perform Grayscale Conversion on CPU.
*/
#ifndef CPU_GRAYSCALE_H
#define CPU_GRAYSCALE_H

#include "image/image.h"

namespace cpu {

    /**
     * Converts an image to grayscale using the CPU.
     *
     * @param input The input image of type Image::Image (must be 1 or 3 channels).
     * @return A grayscale version of the input image.
     */
    Image grayscale(const Image& input);

} // namespace cpu

#endif // CPU_GRAYSCALE_H