#ifndef CPU_GRAYSCALE_H
#define CPU_GRAYSCALE_H

#include "image/image.h"
#include <memory>

namespace fast_img_proc {
    namespace cpu {

        /**
         * Converts an image to grayscale using the CPU.
         *
         * @param input The input image.
         * @return A grayscale version of the input image.
         */
        Image grayscale(const Image& input);

    } // namespace cpu
} // namespace fast_img_proc

#endif // CPU_GRAYSCALE_H