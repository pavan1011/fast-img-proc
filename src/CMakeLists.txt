# Add logging module first since other modules depend on it
add_subdirectory(logging)

# Main fast-img-proc library
add_library(image_lib
    image/image.cpp
    cpu/grayscale.cpp
    cpu/hist_equalize.cpp
    cpu/gauss_blur.cpp
    cpu/sobel_edge_detect.cpp
    processing/processor.cpp
)

target_include_directories(image_lib
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/external/stb
)

target_link_libraries(image_lib 
    PRIVATE 
        TBB::tbb
        project_options
        logging
)

if(USE_CUDA)
    target_sources(image_lib PRIVATE
        gpu/grayscale.cu
        gpu/sobel_edge_detect.cu
    )
    set_source_files_properties(
        gpu/grayscale.cu
        gpu/sobel_edge_detect.cu
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
    )
    set_target_properties(image_processing_cuda PROPERTIES
        // Allow function calls between .cu files
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    target_compile_features(image_lib PUBLIC cuda_std_20)
endif()