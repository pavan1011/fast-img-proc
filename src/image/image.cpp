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

// Fix max width and height of images
constexpr const int MAX_WH = UINT32_MAX/2;

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

Image::Image(uint32_t width, uint32_t height, uint8_t channels) {
    
    // Input Validation for channels
    if(channels != 1 && channels != 3){
        LOG(ERROR, "Failed to create image: Only grayscale and RGB supported. "
                    "No. of channels must be 1 or 3.");
        throw std::invalid_argument("No. of channels must be 1 or 3.");
    }
    if (width > MAX_WH || height > MAX_WH){
        LOG(ERROR, "Failed to create image with dimensions width: {} x height: {} ",
                    width, height);
        LOG(ERROR, "Max supported height or width : {}", MAX_WH);
        throw std::invalid_argument("width or height too large");
    }
    m_width = width;
    m_height = height;
    m_channels = channels;
    LOG(DEBUG, "Constructing Image() from width, height, channels");
    LOG(DEBUG, "wdith:{}, height:{}, channels:{}", width, height, channels);
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