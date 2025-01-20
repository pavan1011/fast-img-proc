#include <gtest/gtest.h>
#include "image/image.h"
#include <memory>

class ImageTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create 2x2 RGB test image
        rgb_image = std::make_unique<Image>(2, 2, 3);
        auto* data = rgb_image->data();
        // Red, Green, Blue, White pixels
        data[0] = 255; data[1] = 0;   data[2] = 0;    // Red
        data[3] = 0;   data[4] = 255; data[5] = 0;    // Green
        data[6] = 0;   data[7] = 0;   data[8] = 255;  // Blue
        data[9] = 255; data[10] = 255; data[11] = 255; // White

        // Create 2x2 grayscale test image
        gray_image = std::make_unique<Image>(2, 2, 1);
        data = gray_image->data();
        data[0] = 100; // Top-left
        data[1] = 150; // Top-right
        data[2] = 200; // Bottom-left
        data[3] = 250; // Bottom-right
    }

    std::unique_ptr<Image> rgb_image;
    std::unique_ptr<Image> gray_image;
};

TEST_F(ImageTest, Construction) {
    EXPECT_NO_THROW(Image(100, 100, 1));
    EXPECT_NO_THROW(Image(100, 100, 3));
    
    // Test invalid dimensions
    EXPECT_THROW(Image(static_cast<uint32_t>(-1), 100, 1), std::invalid_argument);
    EXPECT_THROW(Image(100, static_cast<uint32_t>(-1), 1), std::invalid_argument);
    
    // Test invalid channel count
    EXPECT_THROW(Image(100, 100, -1), std::invalid_argument);
}

TEST_F(ImageTest, Dimensions) {
    EXPECT_EQ(rgb_image->width(), 2);
    EXPECT_EQ(rgb_image->height(), 2);
    EXPECT_EQ(rgb_image->channels(), 3);
    
    EXPECT_EQ(gray_image->width(), 2);
    EXPECT_EQ(gray_image->height(), 2);
    EXPECT_EQ(gray_image->channels(), 1);
}

TEST_F(ImageTest, DataAccess) {
    auto* rgb_data = rgb_image->data();
    EXPECT_EQ(rgb_data[0], 255);  // Red component of first pixel
    EXPECT_EQ(rgb_data[4], 255);  // Green component of second pixel
    EXPECT_EQ(rgb_data[8], 255);  // Blue component of third pixel
    
    auto* gray_data = gray_image->data();
    EXPECT_EQ(gray_data[0], 100);  // Top-left pixel
    EXPECT_EQ(gray_data[3], 250);  // Bottom-right pixel
}

TEST_F(ImageTest, DataModification) {
    auto* data = rgb_image->data();
    data[0] = 128;  // Modify red component of first pixel
    EXPECT_EQ(rgb_image->data()[0], 128);
}

TEST_F(ImageTest, ImageSave) {
    // Test PNG save
    EXPECT_TRUE(rgb_image->save("test_rgb_save_cpptest.png"));
    EXPECT_TRUE(gray_image->save("test_gray_save_cpptest.png"));
    
    // Test JPEG save
    EXPECT_TRUE(rgb_image->save("test_rgb_save_cpptest.jpg"));
    EXPECT_TRUE(gray_image->save("test_gray_save_cpptest.jpg"));
    
    // Test invalid extension
    EXPECT_FALSE(rgb_image->save("test.invalid"));
}