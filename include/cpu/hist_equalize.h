/**
 * @file hist_equalize.h
 * @brief Interface to perform Histogram Equalization on CPU.
*/
#ifndef HIST_EQUALIZE_H
#define HIST_EQUALIZE_H
#include "image/image.h"

/**
 * @namespace cpu
 * @brief Contains CPU implementations of image processing algorithms
 */
namespace cpu {

    /**
     * @brief Perform histogram equalization on an image
     * @param input The input image of type Image::Image (must be 1 or 3 channels)
     * @return Equalized image with same dimensions and channels as input
     * @throws std::runtime_error if input image has invalid number of channels
     * 
     * For grayscale images (1 channel): directly equalizes the intensity
     * For RGB images (3 channels): converts to YCrCb, equalizes Y component, converts back to RGB
     */
    Image equalize_histogram(const Image& input);

    /**
     * @brief Convert RGB to YCrCb color space
     * @param input RGB image to convert
     * @return Image in YCrCb color space
     * 
     * Uses ITU-R BT.601 standard with the following formulas:
     * - Y  =  0.299R + 0.587G + 0.114B
     * - Cr = 128 + (R-Y) * 0.713  [0.713 = 0.5/(1-0.299)]
     * - Cb = 128 + (B-Y) * 0.564  [0.564 = 0.5/(1-0.114)]
     */
    Image rgb_to_ycrcb(const Image& input);
    
    /**
     * @brief Convert YCrCb back to RGB color space
     * @param input YCrCb image to convert
     * @return Image in RGB color space
     * 
     * Conversion formulas:
     * - R = Y + 1.403Cr  [1.403 = 1/0.713]
     * - G = Y - 0.344Cb - 0.714Cr
     * - B = Y + 1.773Cb  [1.773 = 1/0.564]
     */
    Image ycrcb_to_rgb(const Image& input);
    
    /**
     * @brief Equalize single channel using histogram equalization
     * @param data Pointer to image data
     * @param size Number of pixels to process
     * @param stride Step size between pixels (default=1)
     * 
     * Performs histogram equalization on a single channel of image data.
     * The stride parameter allows processing interleaved data by skipping bytes.
     */
    void equalize_channel(unsigned char* data, size_t size, size_t stride = 1);

} //namespace cpu

#endif // HIST_EQUALIZE_H
