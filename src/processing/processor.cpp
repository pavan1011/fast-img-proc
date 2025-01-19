#include "processing/processor.h"
#include "cpu/grayscale.h"
#include "cpu/hist_equalize.h"
#include "cpu/gauss_blur.h"
#include "cpu/sobel_edge_detect.h"
#include "logging/logging.h"
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
            LOG(WARN, "GPU requested but not available. Falling back to CPU");
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
                    LOG(ERROR, "Error in grayscale operation! GPU processing failed {}:");
                    LOG(WARN, "Falling back to CPU implementation.", e.what());
                    return cpu::grayscale(input);
                }
                #else
                LOG(WARN, "GPU support not compiled. Using CPU."
                          "Falling back to CPU implementation.");
                return cpu::grayscale(input);
                #endif
                
            default:
                // TODO: Remove throw and return error code instead.
                LOG(ERROR, "Grayscale: invalid hardware option."
                           "Supported: Hardware::CPU, Hardware::GPU, Hardware::AUTO");
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
                LOG(WARN, "GPU support not yet enabled for this operation."
                        " Using CPU instead");
                return cpu::equalize_histogram(input);

            default:
                // TODO: Remove throw and return error code instead.
                LOG(ERROR, 
                    "Equalize Histogram: invalid hardware option."
                    "Supported: Hardware::CPU, Hardware::GPU, Hardware::AUTO");
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
                LOG(WARN, "GPU support not yet enabled for this operation."
                        " Using CPU instead");
                return cpu::gaussian_blur(input, kernel_size, sigma);

            default:
                // TODO: Remove throw and return error code instead.
                LOG(ERROR, "Blur: invalid hardware option."
                     "Supported: Hardware::CPU, Hardware::GPU, Hardware::AUTO");
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
                    LOG(WARN, "GPU processing failed:{} ", e.what());
                    LOG(WARN," Using CPU instead");
                    return cpu::sobel_edge_detect(input, dx, dy, kernel_size);
                }
                #else
                LOG(WARN, "GPU support not compiled. Using CPU.");
                return cpu::sobel_edge_detect(input, dx, dy, kernel_size);
                #endif
                return cpu::sobel_edge_detect(input, dx, dy, kernel_size);

            default:
                // TODO: Remove throw and return error code instead.
                LOG(ERROR, "Edge Detection: Invalid hardware option."
                           "Supported: Hardware::CPU, Hardware::GPU, Hardware::AUTO");
                throw std::runtime_error("Invalid hardware option");
        }
    }
} // namespace processing