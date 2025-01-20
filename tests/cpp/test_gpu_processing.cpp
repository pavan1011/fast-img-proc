#include <gtest/gtest.h>
#include "image/image.h"
#include "gpu/sobel_edge_detect.cuh"

class GPUInternalTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Similar setup as CPUInternal Test
        rgb_image = std::make_unique<Image>(2, 2, 3);
        auto* data = rgb_image->data();
        // Red, Green, Blue, White pixels
        data[0] = 255; data[1] = 0;   data[2] = 0;    // Red
        data[3] = 0;   data[4] = 255; data[5] = 0;    // Green
        data[6] = 0;   data[7] = 0;   data[8] = 255;  // Blue
        data[9] = 255; data[10] = 255; data[11] = 255; // White

        // Create 2x2 grayscale test image
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

TEST_F(GPUInternalTest, SobelEdgeDetection) {
    // Test invalid parameters
    EXPECT_THROW(gpu::sobel_edge_detect(*gray_image, -1, 0, 3), 
                 std::invalid_argument);
    EXPECT_THROW(gpu::sobel_edge_detect(*gray_image, 0, 0, 3), 
                 std::invalid_argument);
    EXPECT_THROW(gpu::sobel_edge_detect(*gray_image, 1, 1, 2), 
                 std::invalid_argument);
    
    // Test X gradient
    auto result_x = gpu::sobel_edge_detect(*gray_image, 1, 0, 3);
    ASSERT_EQ(result_x.channels(), 1);
    ASSERT_EQ(result_x.width(), gray_image->width());
    ASSERT_EQ(result_x.height(), gray_image->height());
    
    // Test Y gradient
    auto result_y = gpu::sobel_edge_detect(*gray_image, 0, 1, 3);
    ASSERT_EQ(result_y.channels(), 1);
    
    // Test combined XY gradient
    auto result_xy = gpu::sobel_edge_detect(*gray_image, 1, 1, 3);
    ASSERT_EQ(result_xy.channels(), 1);
    
    // Test RGB input validation
    EXPECT_THROW(gpu::sobel_edge_detect(*rgb_image, 1, 0, 3), 
                 std::invalid_argument);
}