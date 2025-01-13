#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <memory>
#include <vector>

class Image {
public:
    Image(const std::string& filepath);
    Image(uint32_t width, uint32_t height, uint8_t channels);
    ~Image();

    // Enable move
    Image(Image&& other) noexcept = default;
    Image& operator=(Image&& other) noexcept = default;
    
    // Delete copy
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