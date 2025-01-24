/**
 * @file main.cpp
 * @brief Example usage of image processing operations on CPU and GPU
 */

#include "image/image.h"
#include "processing/processor.h"
#include <iostream>
#include <string>

void print_usage() {
    std::cout << "Usage: ./fast_img_proc <input_image> <output_image> <operation> <backend>\n\n"
              << "Operations:\n"
              << "  grayscale        - Convert to grayscale\n"
              << "  blur <size> <sigma>    - Gaussian blur (size=3,5,7, sigma>0)\n"
              << "  edge <dx> <dy> <size>  - Sobel edge detection (dx,dy=0,1, size=3,5,7)\n"
              << "  equalize         - Histogram equalization\n\n"
              << "Backend:\n"
              << "  auto - Automatic hardware selection\n"
              << "  cpu  - Force CPU processing\n"
              << "  gpu  - Force GPU processing (falls back to CPU if unavailable)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage();
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    std::string operation = argv[3];
    std::string backend = argv[4];

    // Parse output path for different hardware results
    std::string::size_type const p(outputPath.find_last_of('.'));
    const auto ext = outputPath.substr(outputPath.find_last_of('.') + 1);
    std::string output_filename_noext = outputPath.substr(0, p);
    std::string output_path_auto = output_filename_noext + "_auto." + ext;
    std::string output_path_cpu = output_filename_noext + "_cpu." + ext;
    std::string output_path_gpu = output_filename_noext + "_gpu." + ext;

    try {
        Image input_img(inputPath);
        std::cout << "Loaded image: " << input_img.width() << "x" << input_img.height() 
                 << " channels: " << input_img.channels() << "\n";

        if (operation == "grayscale") {
            std::cout << "Converting to grayscale...\n";
            if (backend == "auto"){
                processing::grayscale(input_img).save(output_path_auto);
            }else if (backend == "cpu")
            {
                processing::grayscale(input_img, processing::Hardware::CPU).save(output_path_cpu);
            }else if (backend == "gpu")
            {
                processing::grayscale(input_img, processing::Hardware::GPU).save(output_path_gpu);        
            }else{
                print_usage();
            }

        }
        else if (operation == "blur") {
            if (argc < 6) {
                std::cout << "Usage: ... blur <kernel_size> <sigma>\n";
                return 1;
            }
            int kernel_size = std::stoi(argv[4]);
            float sigma = std::stof(argv[5]);
            std::cout << "Applying Gaussian blur (size=" << kernel_size << ", sigma=" << sigma << ")...\n";
            
            if (backend == "auto"){
                processing::blur(input_img, kernel_size, sigma).save(output_path_auto);
            }else if (backend == "cpu")
            {
                processing::blur(input_img, kernel_size, sigma, processing::Hardware::CPU).save(output_path_cpu);
            }else if (backend == "gpu")
            {
                processing::blur(input_img, kernel_size, sigma, processing::Hardware::GPU).save(output_path_gpu);
            }else{
                print_usage();
            }
        }
        else if (operation == "edge") {
            if (argc < 7) {
                std::cout << "Usage: ... edge <dx> <dy> <kernel_size>\n";
                return 1;
            }
            int dx = std::stoi(argv[4]);
            int dy = std::stoi(argv[5]);
            int kernel_size = std::stoi(argv[6]);
            std::cout << "Detecting edges (dx=" << dx << ", dy=" << dy 
                     << ", size=" << kernel_size << ")...\n";
            
            if (backend == "auto"){
                processing::edge_detect(input_img, dx, dy, kernel_size).save(output_path_auto);
            }else if (backend == "cpu")
            {
                processing::edge_detect(input_img, dx, dy, kernel_size, 
                                 processing::Hardware::CPU).save(output_path_cpu);
            }else if (backend == "gpu")
            {
                processing::edge_detect(input_img, dx, dy, kernel_size, 
                                 processing::Hardware::GPU).save(output_path_gpu);
            }else{
                print_usage();
            }
            
        }
        else if (operation == "equalize") {
            std::cout << "Equalizing histogram...\n";

            if (backend == "auto"){
                processing::equalize_histogram(input_img).save(output_path_auto);
            }else if (backend == "cpu")
            {
                processing::equalize_histogram(input_img, 
                                        processing::Hardware::CPU).save(output_path_cpu);
            }else if (backend == "gpu")
            {
                processing::equalize_histogram(input_img, 
                                        processing::Hardware::GPU).save(output_path_gpu);
            }else{
                print_usage();
            }
            
        }
        else {
            std::cerr << "Unknown operation: " << operation << "\n";
            print_usage();
            return 1;
        }

        std::cout << "Processing complete. Output files:\n";
        if (backend == "auto"){
            std::cout << "  Auto: " << output_path_auto << "\n";
        }else if (backend == "cpu")
        {
            std::cout << "  CPU:  " << output_path_cpu << "\n";

        }else if (backend == "gpu")
        {
            std::cout << "  GPU:  " << output_path_gpu << "\n";
        }else{
            print_usage();
        }
                
                 
                 
        
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
