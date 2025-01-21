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

# Set macOS specific properties
if(APPLE)
    set_target_properties(image_lib PROPERTIES
        MACOSX_RPATH ON
        INSTALL_RPATH "@loader_path;@loader_path/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

if(USE_CUDA)
    # Check for Apple Silicon before adding CUDA sources
    if(APPLE)
        execute_process(
            COMMAND uname -m
            OUTPUT_VARIABLE ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT ARCH STREQUAL "arm64")
            target_sources(image_lib 
                PRIVATE
                    gpu/sobel_edge_detect.cu
            )
            set_source_files_properties(
                gpu/sobel_edge_detect.cu
                PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
            )
            set_target_properties(image_lib PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_STANDARD 20
                CUDA_STANDARD_REQUIRED ON
            )
            target_link_libraries(image_lib 
                PRIVATE 
                    project_options
                    logging
            )
            target_compile_features(image_lib PUBLIC cuda_std_20)
        endif()
    else()
        target_sources(image_lib 
            PRIVATE
                gpu/sobel_edge_detect.cu
        )
        set_source_files_properties(
            gpu/sobel_edge_detect.cu
            PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
        )
        set_target_properties(image_lib PROPERTIES
            # Allow function calls between .cu files
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )
        target_link_libraries(image_lib 
            PRIVATE 
                project_options
                logging
        )

        target_compile_features(image_lib PUBLIC cuda_std_20)
    endif()
endif()