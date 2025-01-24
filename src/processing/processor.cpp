/**
 * @file processor.cpp
 * @brief Implementation of hardware-agnostic image processing interface
 */
#include "processing/processor.h"
#include "cpu/grayscale.h"
#include "cpu/hist_equalize.h"
#include "cpu/gauss_blur.h"
#include "cpu/sobel_edge_detect.h"
#include "logging/logging.h"
#include "gpu/sobel_edge_detect.cuh"
#include <iostream>

namespace processing {
    /**
     * @brief Checks if GPU processing is available
     * @return true if CUDA is enabled and GPU is available
     * 
     * Checks both compile-time CUDA support and runtime GPU availability
     */
    bool is_gpu_available() {
        #ifdef USE_CUDA
        return gpu::is_available();
        #else
        return false;
        #endif
    }

    /**
     * @brief Gets currently active processing hardware
     * @return Hardware::GPU if available, otherwise Hardware::CPU
     */
    Hardware get_active_hardware() {
        return is_gpu_available() ? Hardware::GPU : Hardware::CPU;
    }

    /**
     * @brief Resolves requested hardware to actually available hardware
     * @param requested The requested hardware configuration
     * @return Resolved hardware (may fall back to CPU if GPU unavailable)
     * 
     * Resolution rules:
     * - AUTO: Uses best available hardware
     * - GPU: Falls back to CPU if GPU unavailable
     * - CPU: Always honored
     */
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

    /**
     * @brief Converts image to grayscale using specified hardware
     * @param input Source image
     * @param hardware Target processing hardware
     * @return Grayscale image
     * @throws std::runtime_error if invalid hardware specified
     * @note GPU implementation not yet available, falls back to CPU
     */
    Image grayscale(const Image& input, Hardware hardware) {
        hardware = resolve_hardware(hardware);
        
        switch(hardware) {
            case Hardware::CPU:
                return cpu::grayscale(input);

            case Hardware::GPU:
                // TODO: Enable CUDA implementation
                LOG(WARN, "GPU support not yet enabled for this operation."
                        " Using CPU instead");
                return cpu::grayscale(input);

            default:
                // TODO: Remove throw and return error code instead.
                LOG(ERROR, 
                    "Equalize Histogram: invalid hardware option."
                    "Supported: Hardware::CPU, Hardware::GPU, Hardware::AUTO");
                throw std::runtime_error("Invalid hardware option");
        }
    }

    /**
     * @brief Performs histogram equalization using specified hardware
     * @param input Source image
     * @param hardware Target processing hardware
     * @return Equalized image
     * @throws std::runtime_error if invalid hardware specified
     * @note GPU implementation not yet available, falls back to CPU
     */
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

    /**
     * @brief Applies Gaussian blur using specified hardware
     * @param input Source image
     * @param kernel_size Blur kernel size
     * @param sigma Gaussian standard deviation
     * @param hardware Target processing hardware
     * @return Blurred image
     * @throws std::runtime_error if invalid hardware specified
     * @note GPU implementation not yet available, falls back to CPU
     */
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

    /**
     * @brief Performs Sobel edge detection using specified hardware
     * @param input Source image
     * @param dx X derivative order
     * @param dy Y derivative order
     * @param kernel_size Sobel kernel size
     * @param hardware Target processing hardware
     * @return Edge detection result
     * @throws std::runtime_error if invalid hardware specified
     * 
     * GPU processing behavior:
     * - Attempts GPU processing if CUDA is enabled
     * - Falls back to CPU on GPU processing failure
     * - Falls back to CPU if CUDA not compiled
     */
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
                    std::string gpu_error =  e.what();
                    LOG(WARN, "GPU processing failed:{} ", gpu_error);
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