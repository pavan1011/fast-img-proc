#include "processing/processor.h"
#include "cpu/grayscale.h"
#include "gpu/grayscale.cuh"
#include <iostream>

namespace processing {
    bool is_gpu_available() {
        #ifdef USE_CUDA
        return gpu::is_available();
        #else
        return false;
        #endif
    }

    Hardware get_active_hardware() {
        return is_gpu_available() ? Hardware::GPU : Hardware::CPU;
    }

    Hardware resolve_hardware(Hardware requested) {
        if (requested == Hardware::AUTO) {
            return get_active_hardware();
        }
        
        if (requested == Hardware::GPU && !is_gpu_available()) {
            std::cerr << "Warning: GPU requested but not available. Falling back to CPU" << std::endl;
            return Hardware::CPU;
        }
        
        return requested;
    }

    Image grayscale(const Image& input, Hardware hardware) {
        hardware = resolve_hardware(hardware);
        
        switch (hardware) {
            case Hardware::CPU:
                return cpu::grayscale(input);
                
            case Hardware::GPU:
                #ifdef USE_CUDA
                try {
                    return gpu::grayscale(input);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: GPU processing failed: " << e.what() 
                              << ". Falling back to CPU." << std::endl;
                    return cpu::grayscale(input);
                }
                #else
                std::cerr << "Warning: GPU support not compiled. Using CPU.\n";
                return cpu::grayscale(input);
                #endif
                
            default:
                // TODO: Remove throw and return error code instead. Pass output image by reference to grayscale().
                throw std::runtime_error("Invalid hardware option");
        }
    }
}