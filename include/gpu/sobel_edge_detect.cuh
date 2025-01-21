/**
 * @file sobel_edge_detect.cuh
 * @brief Interface to perform Sobel Edge Detection GPU.
*/
#ifndef GPU_SOBEL_EDGE_DETECT_H
#define GPU_SOBEL_EDGE_DETECT_H

#include "image/image.h"

/**
 * @namespace gpu
 * @brief GPU-accelerated image processing operations
 */
namespace gpu {
    /**
     * 
     * Sobel edge detection with configurable parameters
     * 
     * @param input The input image.
     * @param dx: derivative order in the X direction
     *        detects edges in x direction if = 1
     * @param dy: derivative order in the Y direction
     *        detects edges in x direction if = 1
     * @param kernel_size: size of the kernel (1, 3, 5, or 7)
     * @return Image after applying appropriate sobel filters.
     */
    Image sobel_edge_detect(const Image& input, 
                          int dx = 1, 
                          int dy = 1,
                          int kernel_size = 3);
    
    // Returns True if GPU is available to use
    bool is_available();

}

#endif // GPU_SOBEL_EDGE_DETECT_H