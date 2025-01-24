#include <gtest/gtest.h>
#include "image/image.h"
#include "cpu/grayscale.h"
#include "cpu/gauss_blur.h"
#include "cpu/hist_equalize.h"
#include "cpu/sobel_edge_detect.h"
#include <memory>
#include <iostream>

class CPUInternalTest : public ::testing::Test {
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

        // Create 3x3 grayscale test image
        gray_image = std::make_unique<Image>(3, 3, 1);
        data = gray_image->data();
        data[0] = 100; // Top-left
        data[1] = 150; // Top-center
        data[2] = 200; // Top-right
        data[3] = 250; // Center-left
        data[4] = 100; // Center
        data[5] = 150; // Center-right
        data[6] = 200; // Bottom-left
        data[7] = 250; // Bottom-center
        data[8] = 100; // Bottom-right
    }

    std::unique_ptr<Image> rgb_image;
    std::unique_ptr<Image> gray_image;
};

TEST_F(CPUInternalTest, GrayscaleConversion) {
    auto result = cpu::grayscale(*rgb_image);
    ASSERT_EQ(result.channels(), 1);
    ASSERT_EQ(result.width(), rgb_image->width());
    ASSERT_EQ(result.height(), rgb_image->height());
    
    auto* data = result.data();
    // Check RGB to grayscale conversion using known weights
    EXPECT_NEAR(data[0], 0.299f * 255, 1);  // Red pixel
    EXPECT_NEAR(data[1], 0.587f * 255, 1);  // Green pixel
    EXPECT_NEAR(data[2], 0.114f * 255, 1);  // Blue pixel
    EXPECT_NEAR(data[3], 255, 1);           // White pixel
}

TEST_F(CPUInternalTest, GrayscaleInputValidation) {  
    // Test already grayscale image
    auto result = cpu::grayscale(*gray_image);
    EXPECT_EQ(result.channels(), 1);
}

TEST_F(CPUInternalTest, GaussBlur) {
    auto result = cpu::gaussian_blur(*gray_image, 3);  // 3x3 kernel
    ASSERT_EQ(result.channels(), 1);
    ASSERT_EQ(result.width(), gray_image->width());
    ASSERT_EQ(result.height(), gray_image->height());
    
    // Test kernel size validation
    // Even kernel size
    EXPECT_THROW(cpu::gaussian_blur(*gray_image, 4, 0), std::invalid_argument);
    // Negative kernel size
    EXPECT_THROW(cpu::gaussian_blur(*gray_image, -1, 0), std::invalid_argument);
}

TEST_F(CPUInternalTest, HistogramEqualization) {
    auto result_gray = cpu::equalize_histogram(*gray_image);
    ASSERT_EQ(result_gray.channels(), 1);
    ASSERT_EQ(result_gray.width(), gray_image->width());
    ASSERT_EQ(result_gray.height(), gray_image->height());
    
    auto result_rgb = cpu::equalize_histogram(*rgb_image);
    // Test RGB input validation
    ASSERT_EQ(result_rgb.channels(), 3);
    ASSERT_EQ(result_rgb.width(), rgb_image->width());
    ASSERT_EQ(result_rgb.height(), rgb_image->height());
}

TEST_F(CPUInternalTest, SobelEdgeDetection) {
    // Test invalid parameters
    EXPECT_THROW(cpu::sobel_edge_detect(*gray_image, -1, 0, 3), 
                 std::invalid_argument);
    EXPECT_THROW(cpu::sobel_edge_detect(*gray_image, 0, 0, 3), 
                 std::invalid_argument);
    EXPECT_THROW(cpu::sobel_edge_detect(*gray_image, 1, 1, 2), 
                 std::invalid_argument);
    
    // Test X gradient
    auto result_x = cpu::sobel_edge_detect(*gray_image, 1, 0, 3);
    ASSERT_EQ(result_x.channels(), 1);
    ASSERT_EQ(result_x.width(), gray_image->width());
    ASSERT_EQ(result_x.height(), gray_image->height());
    
    // Test Y gradient
    auto result_y = cpu::sobel_edge_detect(*gray_image, 0, 1, 3);
    ASSERT_EQ(result_y.channels(), 1);
    
    // Test combined XY gradient
    auto result_xy = cpu::sobel_edge_detect(*gray_image, 1, 1, 3);
    ASSERT_EQ(result_xy.channels(), 1);
    
    // Test RGB input validation
    EXPECT_THROW(cpu::sobel_edge_detect(*rgb_image, 1, 0, 3), 
                 std::invalid_argument);
}