/**
 * @file hist_equalize.cpp
 * @brief CPU implementation of histogram equalization and color space conversions
 */
#include "cpu/hist_equalize.h"
#include <logging/logging.h>
#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <iostream>
#include <numeric>

namespace cpu {
    /**
     * @brief Converts RGB image to YCrCb color space
     * 
     * Uses ITU-R BT.601 standard with parallel processing:
     * - Y  = 0.299R + 0.587G + 0.114B
     * - Cr = 128 + (R-Y) * 0.713
     * - Cb = 128 + (B-Y) * 0.564
     * 
     * @param input RGB image with 3 channels
     * @return YCrCb image with same dimensions
     * @throws std::runtime_error if input doesn't have 3 channels
     */
    Image rgb_to_ycrcb(const Image& input) {
        if (input.channels() != 3) {
            LOG(ERROR, "Only 3 channels (RGB) supported as color input image for "
                       "equalizing histogram.");
            throw std::runtime_error("RGB to YCrCb conversion requires 3 channels");
        }

        Image output(input.width(), input.height(), 3);
        const auto size = input.width() * input.height();
        const auto in_data = input.data();
        const auto out_data = output.data();

        std::vector<int> indices(size);
        std::iota(indices.begin(), indices.end(), 0);

        // Convert pixels in parallel. STL makes calls to TBB library
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [in_data, out_data](int i) {
                const int pixel = i * 3;
                const float r = static_cast<float>(in_data[pixel]);
                const float g = static_cast<float>(in_data[pixel + 1]);
                const float b = static_cast<float>(in_data[pixel + 2]);

                // Conversion as per ITU-R BT.601
                const float y  = 0.299f * r + 0.587f * g + 0.114f * b;
                const float cr = 128.0f + (r - y) * 0.713f;  // 0.713 = 0.5/(1-0.299)
                const float cb = 128.0f + (b - y) * 0.564f;  // 0.564 = 0.5/(1-0.114)

                out_data[pixel] = static_cast<unsigned char>(std::clamp(y, 0.0f, 255.0f));
                out_data[pixel + 1] = static_cast<unsigned char>(std::clamp(cr, 0.0f, 255.0f));
                out_data[pixel + 2] = static_cast<unsigned char>(std::clamp(cb, 0.0f, 255.0f));
            });
        LOG(DEBUG, "RGB to YCrCb done.");
        return output;
    }

    /**
     * @brief Converts YCrCb image back to RGB color space
     * 
     * Inverse BT.601 conversion with parallel processing:
     * - R = Y + 1.403Cr
     * - G = Y - 0.344Cb - 0.714Cr
     * - B = Y + 1.773Cb
     * 
     * @param input YCrCb image with 3 channels
     * @return RGB image with same dimensions
     * @throws std::runtime_error if input doesn't have 3 channels
     */
    Image ycrcb_to_rgb(const Image& input) {
        if (input.channels() != 3) {
            LOG(ERROR, "Only 3 channels (RGB) supported as color input image for "
                        "equalizing histogram.");
            throw std::runtime_error("YCrCb to RGB conversion requires 3 channels");
        }

        Image output(input.width(), input.height(), 3);
        const auto size = input.width() * input.height();
        const auto in_data = input.data();
        const auto out_data = output.data();

        std::vector<int> indices(size);
        std::iota(indices.begin(), indices.end(), 0);

        // Convert pixels in parallel. STL makes calls to TBB library
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [in_data, out_data](int i) {
                const int pixel = i * 3;
                const float y  = static_cast<float>(in_data[pixel]);
                const float cr = static_cast<float>(in_data[pixel + 1]) - 128.0f;
                const float cb = static_cast<float>(in_data[pixel + 2]) - 128.0f;

                // Inverse BT.601 conversion
                const float r = y + 1.403f * cr;           // 1.403 = 1/0.713
                const float g = y - 0.344f * cb - 0.714f * cr;
                const float b = y + 1.773f * cb;           // 1.773 = 1/0.564

                out_data[pixel] = static_cast<unsigned char>(std::clamp(r, 0.0f, 255.0f));
                out_data[pixel + 1] = static_cast<unsigned char>(std::clamp(g, 0.0f, 255.0f));
                out_data[pixel + 2] = static_cast<unsigned char>(std::clamp(b, 0.0f, 255.0f));
            });
        LOG(DEBUG, "YCrCb to RGB done.");
        return output;
    }

    /**
     * @brief Equalizes a single channel using histogram equalization
     * 
     * Implementation steps:
     * 1. Calculate histogram
     * 2. Compute cumulative distribution function (CDF)
     * 3. Find minimum non-zero CDF value
     * 4. Create lookup table using equalization formula
     * 5. Apply lookup table in parallel
     * 
     * @param data Pointer to image data
     * @param size Total number of pixels * stride
     * @param stride Step size between pixels to access pixels from 
     *        individual channels (default=1)
     * @throws std::invalid_argument if invalid parameters provided
     */
    void equalize_channel(unsigned char* data, size_t size, size_t stride) {

        // TODO: Replace throw with returning error code.
        if (stride == 0) {
            LOG(ERROR, "Invalid argument passed to cpu::equalize_channel():"
                            "Stride cannot be zero");
            throw std::invalid_argument("Stride cannot be zero");
        }
        if (data == nullptr) {
            LOG(ERROR, "Invalid argument passed to cpu::equalize_channel():"
                           "Stride cannot be zero");
            throw std::invalid_argument("Data pointer cannot be null");
        }
        if (size == 0) {
            LOG(ERROR, "Invalid argument passed to cpu::equalize_channel(): ");
            LOG(ERROR, "Image size cannot be zero");
            throw std::invalid_argument("Size cannot be zero");
        }
        if (size % stride != 0) {
            LOG(ERROR, "Invalid argument passed to cpu::equalize_channel(): ");
            LOG(ERROR, "No. of pixels must be divisible by channels! "
                       "Only 3 channels (RGB) supported for color images");
            throw std::invalid_argument("Size must be divisible by stride");
        }

        // Step 1: Calculate histogram
        std::vector<int> histogram(256, 0);
        for (size_t i = 0; i < size; i+=stride) {
            ++histogram[data[i]];
        }

        // Step 2: Calculate cumulative distribution function (CDF)
        std::vector<int> cdf(256, 0);
        cdf[0] = histogram[0];
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + histogram[i];
        }

        // Step 3: Find first non-zero value in CDF (cdf_min)
        int cdf_min = 0;
        for (int i = 0; i < 256; ++i) {
            if (cdf[i] > 0) {
                cdf_min = cdf[i];
                break;
            }
        }

        // Step 4: Create lookup table for equalization
        // Formula: h(v) = round((cdf(v) - cdf_min) * (L-1)/(M*N - cdf_min))
        // where L is number of gray levels (256), M*N is image size
        std::vector<unsigned char> lut(256);
        // scale = (L-1)/(M*N - cdf_min)
        float scale = 255.0f / ((size/stride) - cdf_min);
        for (int i = 0; i < 256; ++i) {
            lut[i] = static_cast<unsigned char>(
                std::round(std::clamp((cdf[i] - cdf_min) * scale, 0.0f, 255.0f))
            );
        }
        
        // Create indices for strided access
        std::vector<size_t> indices(size/stride);
        std::iota(indices.begin(), indices.end(), 0);


        // Step 5: Apply lookup table to image data in parallel
        std::for_each(std::execution::par_unseq, 
            indices.begin(), indices.end(),
            [data, stride, &lut](size_t i) {
                data[i * stride] = lut[data[i * stride]];
            });

        LOG(DEBUG, "Equalize Luma channel done.");
    }

    Image equalize_histogram(const Image& input) {
        LOG(DEBUG, "CPU: Starting histogram equalization.");
        if (input.channels() == 1) {
            // For grayscale images, equalize directly
            Image output(input.width(), input.height(), 1);
            const auto size = input.width() * input.height();
            std::copy_n(input.data(), size, output.data());

            // Call to equalize single channel
            equalize_channel(output.data(), size);
            LOG(DEBUG, "CPU: Histogram equalization done.");
            return output;
        }
        else if (input.channels() == 3) {
            // For RGB images:
            // 1. Convert to YCrCb
            // 2. Equalize Y channel only (preserve color information)
            // 3. Convert back to RGB
            auto ycrcb = rgb_to_ycrcb(input);
            const auto size = input.width() * input.height();
            
            // Equalize Y channel only (first channel)
            // Get pointer to Y channel and stride by 3 to skip Cr and Cb
            equalize_channel(ycrcb.data(), size*3, 3);

            LOG(DEBUG, "CPU: Histogram equalization done.");
            // Convert back to RGB
            return ycrcb_to_rgb(ycrcb);
        }
        else {
            LOG(ERROR, "Image must have 1 or 3 channels (Gray/RGB) for histogram equalization");
            throw std::runtime_error("Image must have 1 or 3 channels for histogram equalization");
        }
    }

} // namespace cpu