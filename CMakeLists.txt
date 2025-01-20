cmake_minimum_required(VERSION 3.18)
project(fast_img_proc 
        VERSION 1.0
        LANGUAGES CXX)

# Configure version header to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/version.h
    @ONLY
)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable -fPIC flag for CMake
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Setting Build Type for logging levels and perf measurement
set(CMAKE_CONFIGURATION_TYPES "Debug;Verbose;Profile;Release" 
    CACHE STRING "Available build types" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(project_options INTERFACE)
target_compile_definitions(project_options INTERFACE
    $<$<CONFIG:Debug>:DEBUG_BUILD>
    $<$<CONFIG:Verbose>:VERBOSE_BUILD>
    $<$<CONFIG:Profile>:PROFILE_BUILD>
)

if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG_BUILD")
    set(CMAKE_CXX_FLAGS_VERBOSE "/O2 /DVERBOSE_BUILD")
    set(CMAKE_CXX_FLAGS_PROFILE "/O2 /DPROFILE_BUILD")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG_BUILD")
    set(CMAKE_CXX_FLAGS_VERBOSE "-O2 -DVERBOSE_BUILD")
    set(CMAKE_CXX_FLAGS_PROFILE "-O2 -DPROFILE_BUILD")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

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

# Find Python and nanobind for bindings
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind)

# Documentation option
option(BUILD_DOCUMENTATION "Build documentation with Doxygen" OFF)

# Add project fast-img-proc/src subdirectory
add_subdirectory(src)

# Documentation configuration
if(BUILD_DOCUMENTATION)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs)
        
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
                      ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    else()
        message(STATUS "Doxygen not found, documentation will not be built")
    endif()
endif()

# Create main executable
add_executable(fast_img_proc main.cpp)
target_link_libraries(fast_img_proc PRIVATE image_lib)
target_include_directories(fast_img_proc 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
)


# Create Python bindings to fast-img-processing
nanobind_add_module(fast_image_processing bindings.cpp)
target_link_libraries(fast_image_processing PRIVATE image_lib)

