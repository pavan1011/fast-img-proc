#include "processing/processor.h"
#include "cpu/grayscale.h"
#include "cpu/hist_equalize.h"
#include "cpu/gauss_blur.h"
#include "cpu/sobel_edge_detect.h"
#include "gpu/grayscale.cuh"
#include "gpu/sobel_edge_detect.cuh"
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
            std::cerr << "Warning: GPU requested but not available. Falling back to CPU\n";
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
                              << ". Falling back to CPU.\n";
                    return cpu::grayscale(input);
                }
                #else
                std::cerr << "Warning: GPU support not compiled. Using CPU.\n";
                return cpu::grayscale(input);
                #endif
                
            default:
                // TODO: Remove throw and return error code instead.
                throw std::runtime_error("Invalid hardware option");
        }
    }

    Image equalize_histogram(const Image& input, Hardware hardware) {
        hardware = resolve_hardware(hardware);

        switch(hardware) {
            case Hardware::CPU:
                return cpu::equalize_histogram(input);

            case Hardware::GPU:
                // TODO: Enable CUDA implementation
                std::cerr << "Warning: GPU support not yet enabled for this operation."
                          << "Using CPU instead.\n";
                return cpu::equalize_histogram(input);

            default:
                // TODO: Remove throw and return error code instead.
                throw std::runtime_error("Invalid hardware option");
        }
    }

    Image blur(const Image& input, int kernel_size, float sigma, Hardware hardware) {
        hardware = resolve_hardware(hardware);

        switch(hardware) {
            case Hardware::CPU:
                return cpu::gaussian_blur(input, kernel_size, sigma);

            case Hardware::GPU:
                // TODO: Enable CUDA implementation
                std::cerr << "Warning: GPU support not yet enabled for this operation."
                          << "Using CPU instead.\n";
                return cpu::gaussian_blur(input, kernel_size, sigma);

            default:
                // TODO: Remove throw and return error code instead.
                throw std::runtime_error("Invalid hardware option");
        }
    }

    Image edge_detect(const Image& input, int dx, int dy,
                            int kernel_size, Hardware hardware) {
        hardware = resolve_hardware(hardware);

        switch(hardware) {
            case Hardware::CPU:
                return cpu::sobel_edge_detect(input, dx, dy, kernel_size);

            case Hardware::GPU:
                #ifdef USE_CUDA
                try {
                    return gpu::sobel_edge_detect(input, dx, dy, kernel_size);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: GPU processing failed: " << e.what() 
                              << ". Falling back to CPU.\n";
                    return cpu::sobel_edge_detect(input, dx, dy, kernel_size);
                }
                #else
                std::cerr << "Warning: GPU support not compiled. Using CPU.\n";
                return cpu::sobel_edge_detect(input, dx, dy, kernel_size);
                #endif
                return cpu::sobel_edge_detect(input, dx, dy, kernel_size);

            default:
                // TODO: Remove throw and return error code instead.
                throw std::runtime_error("Invalid hardware option");
        }
    }
} // namespace processing