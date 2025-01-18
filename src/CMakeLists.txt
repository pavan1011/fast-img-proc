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

target_link_libraries(image_lib PRIVATE TBB::tbb)

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
    target_compile_features(image_lib PUBLIC cuda_std_17)
endif()