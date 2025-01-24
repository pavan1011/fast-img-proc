/**
 * @file image.h
 * @brief Core Image class definition, used as input to processing functions.
*/

#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <memory>
#include <vector>

/**
 * @class Image
 * @brief A class for handling and manipulating image data
 * 
 * This class provides functionality for managing image data including
 * creation, modification, and basic image processing operations.
 * 
 * The class implements move semantics and prohibits copying to ensure
 * efficient resource management.

 * Images are represented in the RGB colorspace using 8 bits to represent each pixel.
 */
class Image {
public:
    /**
     * @brief Constructs an image from file
     * @param filepath Path to the image file to be loaded
     * @throws std::runtime_error If file cannot be loaded
     */
    Image(const std::string& filepath);

    /**
     * @brief Constructs an empty image with each pixel initialized to 0
     *        from the following dimensional parameters:
     * @param width: length of row (in pixels) 
     * @param height: Length of column (in pixels)
     * @param channels: No. of color channels to represent 
     *                  the image (1=Grayscale, 3=RGB)
     * @throws std::invalid_argument If image doesn't contain either 1 or 3 channels,
     *  or if Height > MAX_H (UINT32_MAX / 4) or Width is > MAX_H (UINT32_MAX / 4).
     */
    Image(uint32_t width, uint32_t height, uint8_t channels);
    ~Image();

    // Enable move
    /**
     * @brief Move constructor
     * @param other The source image to move from
     * 
     * Transfers ownership of resources from the source image to the newly
     * constructed image. The source image will be left in a valid but
     * unspecified state.
     */
    Image(Image&& other) noexcept = default;

    /**
     * @brief Move assignment operator
     * @param other The source image to move from
     * @return Reference to the current image
     * 
     * Transfers ownership of resources from the source image to this image.
     * The source image will be left in a valid but unspecified state.
     */
    Image& operator=(Image&& other) noexcept = default;
 
    /**
     * @brief Copy constructor (deleted)
     * 
     * Copying is disabled to prevent unintended resource duplication
     */
    Image(const Image&) = delete;

    /**
     * @brief Copy assignment operator (deleted)
     * 
     * Copying is disabled to prevent unintended resource duplication
     */
    Image& operator=(const Image&) = delete;

    // Accessors
    /**
     * @brief Get the width of the image
     * @return Width in pixels
     */
    uint32_t width() const { return m_width; }

    /**
     * @brief Get the height of the image
     * @return Height in pixels
     */
    uint32_t height() const { return m_height; }

    /**
     * @brief Get the number of channels in the image
     * @return Number of channels (1 for grayscale, 3 for RGB)
    */
    uint8_t channels() const { return m_channels; }

    /**
     * @brief Get raw pointer to the underlying image data
     * @return const pointer to the raw image data buffer
     * 
     * Returns a pointer to the first byte of the image data array.
     * The data is stored as a contiguous array of unsigned chars where:
     * - For RGB images (3 channels), pixels are stored sequentially as R,G,B values:
     *   [R1,G1,B1, R2,G2,B2, R3,G3,B3, ...] for each pixel
     * - For Grayscale images (1 channel), each byte represents one pixel intensity
     * - Total size of data = width * height * channels
     * 
     * @note The returned pointer remains owned by the Image object
     * @warning Do not delete or modify the returned pointer
     */
    auto data() const noexcept { return m_data.get(); }

    /**
     * @brief Saves the image to a file
     * @param filename Path where the image should be saved
     * @return true if save was successful, false otherwise
     * 
     * Supports common image formats based on file extension:
     * - .png for PNG format
     * - .jpg or .jpeg for JPEG format
     * 
     * The image is saved with the same dimensions and color channels
     * as the source Image object.
     * 
     * @note Directory must exist and be writable
     * @warning Existing files will be overwritten without warning
     */
    bool save(const std::string& filename) const;


private:
    uint32_t m_width{0};
    uint32_t m_height{0};
    uint8_t m_channels{0};
    std::unique_ptr<unsigned char[]> m_data;
};

#endif // IMAGE_H