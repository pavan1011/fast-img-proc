// For stb_image 
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//For stb_image_write
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image/image.h"
#include <stdexcept>
#include <iostream>

namespace fast_img_proc {

Image::Image(const std::string& filepath) {
    int width, height, channels;
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + filepath);
    }
    
    m_width = width;
    m_height = height;
    m_channels = channels;
    m_data = std::unique_ptr<unsigned char[]>(data);
}

Image::Image(int width, int height, int channels)
    : m_width(width), m_height(height), m_channels(channels) {
    m_data = std::make_unique<unsigned char[]>(width * height * channels);
}

Image::~Image() = default;

bool Image::save(const std::string& filepath) const {
    const auto ext = filepath.substr(filepath.find_last_of('.') + 1);
    if (ext == "png") {
        return stbi_write_png(filepath.c_str(), m_width, m_height, m_channels, 
                            m_data.get(), m_width * m_channels) != 0;
    } else if (ext == "jpg" || ext == "jpeg") {
        return stbi_write_jpg(filepath.c_str(), m_width, m_height, m_channels, 
                            m_data.get(), 100) != 0;
    }
    return false;
}

} // namespace fast_img_proc