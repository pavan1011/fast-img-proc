/**
 * @file sobel_edge_detect.cuh
 * @brief Adapter to perform image processing on either CPU or GPU.
*/
#ifndef PROCESSOR_H
#define PROCESSOR_H
#include "image/image.h"
#include <string>

/**
 * @namespace processing
 * @brief Provides hardware-agnostic interface for image processing operations
 */
namespace processing {
    /**
     * @enum Hardware
     * @brief Specifies target hardware for processing operations
     * 
     * @var Hardware::AUTO Automatically selects best available hardware
     * @var Hardware::CPU Forces CPU execution
     * @var Hardware::GPU Forces GPU execution if available, falls back to CPU
     *      implementation if GPU unavailable.
     */
    enum class Hardware {
        AUTO,  // Automatically choose best available
        CPU,
        GPU
    };

    /**
     * @brief Checks if CUDA-capable GPU is available
     * @return true if GPU is available and initialized
     */
    bool is_gpu_available();

    /**
     * @brief Gets currently active processing hardware
     * @return Hardware enum indicating active processor
     */
    Hardware get_active_hardware();

    /**
     * @brief Converts image to grayscale
     * @param input Source image
     * @param hardware Target hardware for processing
     * @return Single-channel grayscale image
     * @throws std::runtime_error if hardware other than AUTO, CPU, or GPU is requested
     */
    Image grayscale(const Image& input, Hardware hardware = Hardware::AUTO);

    /**
     * @brief Performs histogram equalization
     * @param input Source image
     * @param hardware Target hardware for processing
     * @return Equalized image with same dimensions and channels
     * @throws std::runtime_error if hardware other than AUTO, CPU, or GPU is requested
     */
    Image equalize_histogram(const Image& input, Hardware hardware = Hardware::AUTO);

    /**
     * @brief Applies Gaussian blur
     * @param input Source image
     * @param kernel_size Size of Gaussian kernel (must be odd)
     * @param sigma Standard deviation of Gaussian distribution
     * @param hardware Target hardware for processing
     * @return Blurred image with same dimensions and channels
     * @throws std::runtime_error if hardware other than AUTO, CPU, or GPU is requested
     * @throws std::invalid_argument if kernel_size is invalid
     */
    Image blur(const Image& input, int kernel_size = 3, float sigma = 1.0f,
               Hardware hardware = Hardware::AUTO);

    /**
     * @brief Performs Sobel edge detection
     * @param input Source image
     * @param dx Order of X derivative (0 or 1)
     * @param dy Order of Y derivative (0 or 1)
     * @param kernel_size Size of Sobel kernel (1, 3, 5, or 7)
     * @param hardware Target hardware for processing
     * @return Single-channel edge detection result
     * @throws std::runtime_error if hardware other than AUTO, CPU, or GPU is requested
     * @throws std::invalid_argument if parameters are invalid
     * @note Uses kernel_size = 3 if kernel_size = 1 is passed 
     *       (similar to OpenCV implementation)
     */
    Image edge_detect(const Image& input, int dx = 1, int dy = 1,
                            int kernel_size = 3, Hardware hardware = Hardware::AUTO);

} // namespace processing

#endif // PROCESSOR_H