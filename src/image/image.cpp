// For stbi_load
#define STB_IMAGE_IMPLEMENTATION
//For stbi_write
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "image/image.h"
#include "logging/logging.h"
#include <stdexcept>
#include <iostream>

Image::Image(const std::string& filepath) {
    int width, height, channels;
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
    if (!data) {
        LOG(ERROR, "Failed to load image: {}", filepath);
        // TODO: Remove throw and return error code instead.
        throw std::runtime_error("Failed to load image: " + filepath);
    }
    
    // Explicit conversion to correct sized ints
    m_width = static_cast<uint32_t>(width);
    m_height = static_cast<uint32_t>(height);
    // TODO: check if channels is between 0 and 255 before conversion to prevent
    // data-loss/unexpected behavior
    m_channels = static_cast<uint8_t>(channels);
    m_data = std::unique_ptr<unsigned char[]>(data);
}

Image::Image(u_int32_t width, u_int32_t height, u_int8_t channels)
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