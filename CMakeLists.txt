cmake_minimum_required(VERSION 3.18)
project(fast_img_proc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# CUDA Optional
option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
endif()

#Find Thread Building Blocks (TBB) for parallelization
#TODO: Make TBB Optional. Fallback to sequential execution if TBB is not installed.
find_package(TBB REQUIRED)

add_subdirectory(src)

# Create main executable
add_executable(fast_img_proc main.cpp)
target_link_libraries(fast_img_proc PRIVATE image_lib)
target_include_directories(fast_img_proc 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
)