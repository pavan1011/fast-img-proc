/**
 * @file image.h
 * @brief Core image processing class definition
*/

#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <memory>
#include <vector>

/**
 * @class Image
 * @brief Main class for image representation and manipulation
 * 
 * This class provides the core functionality for loading, 
 * representing, and saving images.
 */
class Image {
public:
    /**
     * @brief Constructs an image from file
     * @param filename Path to the image file
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
     * @throws std::invalid_argument If image doesn't c
     */
    Image(uint32_t width, uint32_t height, uint8_t channels);
    ~Image();

    // Enable move
    Image(Image&& other) noexcept = default;
    Image& operator=(Image&& other) noexcept = default;
    
    // Delete copy constructor to prevent accidental copies and memleaks
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    // Accessors
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
    uint8_t channels() const { return m_channels; }
    auto data() const noexcept { return m_data.get(); }

    // Save image to file
    bool save(const std::string& filename) const;


private:
    uint32_t m_width{0};
    uint32_t m_height{0};
    uint8_t m_channels{0};
    std::unique_ptr<unsigned char[]> m_data;
};

#endif // IMAGE_H