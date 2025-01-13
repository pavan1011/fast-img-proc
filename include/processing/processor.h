#ifndef PROCESSOR_H
#define PROCESSOR_H
#include "image/image.h"

namespace processing {
    // Enum to handle different hardware for execution
    enum class Hardware {
        AUTO,  // Automatically choose best available
        CPU,
        GPU
    };

    // Wrapper for grayscale conversion function
    Image grayscale(const Image& input, Hardware hardware = Hardware::AUTO);
    
    // Hardware availability check
    bool is_gpu_available();
    
    // Get currently active hardware
    Hardware get_active_hardware();
}

#endif // PROCESSOR_H