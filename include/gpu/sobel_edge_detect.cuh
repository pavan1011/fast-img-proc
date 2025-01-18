#ifndef GPU_SOBEL_EDGE_DETECT_H
#define GPU_SOBEL_EDGE_DETECT_H

#include "image/image.h"

namespace gpu {
    // Sobel edge detection with configurable parameters
    // dx: detect edges in x direction (0 or 1)
    // dy: detect edges in y direction (0 or 1)
    // kernel_size: size of the kernel (1, 3, 5, or 7)
    Image sobel_edge_detect(const Image& input, 
                          int dx = 1, 
                          int dy = 1,
                          int kernel_size = 3);
}

#endif // GPU_SOBEL_EDGE_DETECT_H