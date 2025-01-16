#include "image/image.h"
#include "processing/processor.h"
#include <iostream>
#include <string>


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
        std::string::size_type const p(outputPath.find_last_of('.'));
        const auto ext = outputPath.substr(outputPath.find_last_of('.') + 1);
        std::string output_filename_noext = outputPath.substr(0, p);
        std::string output_path_auto = output_filename_noext + "_auto." + ext;
        std::string output_path_cpu = output_filename_noext + "_cpu." + ext;
        std::string output_path_gpu = output_filename_noext + "_gpu." + ext;        
        

        std::cout << "Loaded image: " << input_img.width() << "x" << input_img.height() << "\n";

        if (operation == "grayscale") {
            std::cout<<"Operation selected: grayscale." << std::endl;

            // Automatic hardware selection
            auto processed_img_auto = processing::grayscale(input_img);
            processed_img_auto.save(output_path_auto);

            // Force CPU
            auto processed_img_cpu = processing::grayscale(input_img, processing::Hardware::CPU);
            processed_img_cpu.save(output_path_cpu);

            // Try GPU (will fall back to CPU if CUDA unavailable)
            auto processed_img_gpu = processing::grayscale(input_img, processing::Hardware::GPU);
            processed_img_gpu.save(output_path_gpu);

        }
        else if (operation == "equalize")
        {
            std::cout<<"Operation selected: equalize_histogram." << std::endl;

            // Automatic hardware selection
            auto processed_img_auto = processing::equalize_histogram(input_img);
            processed_img_auto.save(output_path_auto);

            // Force CPU
            auto processed_img_cpu = processing::equalize_histogram(input_img, processing::Hardware::CPU);
            processed_img_cpu.save(output_path_cpu);

            // Try GPU (will fall back to CPU if CUDA unavailable)
            auto processed_img_gpu = processing::equalize_histogram(input_img, processing::Hardware::GPU);
            processed_img_gpu.save(output_path_gpu);
        }
        else if (operation == "edge")
        {
            std::cerr << "Coming soon! \n";

        }
        else if (operation == "blur"){
            std::cout<<"Operation selected: blur." << std::endl;

            // Automatic hardware selection
            auto processed_img_auto = processing::blur(input_img);
            processed_img_auto.save(output_path_auto);

            // Force CPU
            auto processed_img_cpu = processing::blur(input_img, 5, 1.4f, processing::Hardware::CPU);
            processed_img_cpu.save(output_path_cpu);

            // Try GPU (will fall back to CPU if CUDA unavailable)
            auto processed_img_gpu = processing::blur(input_img, 5, 1.4f, processing::Hardware::GPU);
            processed_img_gpu.save(output_path_gpu);
        }
        else {
            std::cerr << "Unknown operation: " << operation << "\n";
            std::cerr << "Supported Operations: grayscale, blur, edge, equalize\n";
            return 1;
        }

        
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
