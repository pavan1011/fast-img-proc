#ifndef GPU_GRAYSCALE_CUH
#define GPU_GRAYSCALE_CUH

#include "../image/image.h"

namespace fast_img_proc {
    namespace gpu {

        /**
         * Converts an image to grayscale using CUDA on the GPU.
         *
         * @param input The input image.
         * @return A grayscale version of the input image.
         */
        Image grayscale(const Image& input);
        bool is_available();

    } // namespace gpu
} // namespace fast_img_proc

#endif // GPU_GRAYSCALE_CUH