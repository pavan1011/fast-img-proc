/**
 * @file sobel_edge_detect.cuh
 * @brief Adapter to perform image processing on either CPU or GPU.
*/
#ifndef PROCESSOR_H
#define PROCESSOR_H
#include "image/image.h"
#include <string>

namespace processing {
    // Enum to handle different hardware for execution
    enum class Hardware {
        AUTO,  // Automatically choose best available
        CPU,
        GPU
    };

    // Hardware availability check
    bool is_gpu_available();

    // Get currently active hardware
    Hardware get_active_hardware();

    // Wrapper for grayscale conversion function
    Image grayscale(const Image& input, Hardware hardware = Hardware::AUTO);

    // Wrapper for histogram equalization function
    Image equalize_histogram(const Image& input, Hardware hardware = Hardware::AUTO);

    // Wrapper for gaussian blur function
    Image blur(const Image& input, int kernel_size = 3, float sigma = 1.0f,
               Hardware hardware = Hardware::AUTO);

    // Wrapper for sobel edge detection function
    Image edge_detect(const Image& input, int dx = 1, int dy = 1,
                            int kernel_size = 3, Hardware hardware = Hardware::AUTO);

} // namespace processing

#endif // PROCESSOR_H