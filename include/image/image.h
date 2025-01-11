#ifndef IMAGE_H
#define IMAGE_H

#include <cstdint>
#include <string>
#include <memory>
#include <vector>


namespace fast_img_proc {

class Image {
public:
    Image(const std::string& filepath);
    Image(int width, int height, int channels);
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

} // namespace fast_img_proc

#endif // IMAGE_H