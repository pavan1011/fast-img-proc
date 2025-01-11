#include "image/image.h"
#include "cpu/grayscale.h"
#include "gpu/grayscale.cuh"
#include <iostream>
#include <string>

using namespace fast_img_proc;

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: image_processor <input_image> <output_image> <operation> <backend>\n";
        std::cerr << "Operations: grayscale, blur, edge, equalize\n";
        std::cerr << "Backend: cpu, gpu\n";
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    std::string operation = argv[3];
    std::string backend_str = argv[4];

    try {
        Image input_img(inputPath);
        std::cout << "Loaded image: " << input_img.width() << "x" << input_img.height() << "\n";

        std::cout<<"Before Image processed initialized" << std::endl;

        if (operation == "grayscale") {
            std::cout<<"Operation selected: grayscale." << std::endl;

            if (backend_str == "gpu"){

                #ifdef USE_CUDA
                if (gpu::is_available()) {
                    std::cout << "Using CUDA GPU implementation\n";
                    auto processed_img = gpu::grayscale(input_img);
                    processed_img.save(outputPath);
                }
                #endif 
            }
            else {
                
                    std::cout << "Using CPU implementation\n";
                    auto processed_img = cpu::grayscale(input_img);
                    processed_img.save(outputPath);
                    std::cout << "Processed image saved to " << outputPath << "\n";
            }

        }
        
        else {
            std::cerr << "Unknown operation: " << operation << "\n";
            return 1;
        }

        
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
