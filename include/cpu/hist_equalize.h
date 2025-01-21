/**
 * @file hist_equalize.h
 * @brief Interface to perform Histogram Equalization on CPU.
*/
#ifndef HIST_EQUALIZE_H
#define HIST_EQUALIZE_H
#include "image/image.h"

namespace cpu {

    /**
     * Perform histogram equalization on an image
     * For grayscale images (1 channel): directly equalizes the intensity
     * For RGB images (3 channels): converts to YCrCb, equalizes Y component, converts back to RGB
     * 
     * @param input Input image (must be 1 or 3 channels)
     * @return Equalized image with same dimensions and channels as input
     * @throws std::runtime_error if input image has invalid number of channels
     */
    Image equalize_histogram(const Image& input);

    /**
     * Convert RGB to YCrCb color space using ITU-R BT.601 standard
     * Y  =  0.299R + 0.587G + 0.114B
     * Cr = 128 + (R-Y) * 0.713  [0.713 = 0.5/(1-0.299)]
     * Cb = 128 + (B-Y) * 0.564  [0.564 = 0.5/(1-0.114)]
     */
    Image rgb_to_ycrcb(const Image& input);
    
    /**
     * Convert YCrCb back to RGB color space
     * R = Y + 1.403Cr  [1.403 = 1/0.713]
     * G = Y - 0.344Cb - 0.714Cr
     * B = Y + 1.773Cb  [1.773 = 1/0.564]
     */
    Image ycrcb_to_rgb(const Image& input);
    
    /**
     * Equalize single channel using histogram equalization
     * @param data Pointer to image data
     * @param size Number of pixels to process
     */
    void equalize_channel(unsigned char* data, size_t size, size_t stride = 1);

} //namespace cpu

#endif // HIST_EQUALIZE_H
