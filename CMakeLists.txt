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

# CUDA option: compile with CUDA or no
option(USE_CUDA "Enable CUDA support" OFF)
option(BUILD_DOCUMENTATION "Build documentation with Doxygen" OFF)
option(BUILD_TESTS "Build tests" ON)

# If no build type is defined, use Release as default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Implementing CUDA Optional
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
endif()

# Setting Build Type for logging levels and perf measurement
set(CMAKE_CONFIGURATION_TYPES "Debug;Verbose;Profile;Release" 
    CACHE STRING "Available build types" FORCE)

add_library(project_options INTERFACE)

# Adding various Build options
target_compile_definitions(project_options INTERFACE
    $<$<CONFIG:Debug>:DEBUG_BUILD>
    $<$<CONFIG:Verbose>:VERBOSE_BUILD>
    $<$<CONFIG:Profile>:PROFILE_BUILD>
)

# Debug build: disable optimization (O0), enable source debugging(-g) 
# pass preprocessor macro DEBUG_BUILD
# O2 medium optimization, O3 most optimal.
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG_BUILD")
    set(CMAKE_CXX_FLAGS_VERBOSE "/O2 /DVERBOSE_BUILD")
    set(CMAKE_CXX_FLAGS_PROFILE "/O3 /DPROFILE_BUILD")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG_BUILD")
    set(CMAKE_CXX_FLAGS_VERBOSE "-O2 -DVERBOSE_BUILD")
    set(CMAKE_CXX_FLAGS_PROFILE "-O3 -DPROFILE_BUILD")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

#Find Thread Building Blocks (TBB) for parallelization
#TODO: Make TBB Optional. Fallback to sequential execution if TBB is not installed.
find_package(TBB REQUIRED)

# Find Python and nanobind for bindings
if(DEFINED PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind)


# Add project fast-img-proc/src subdirectory
add_subdirectory(src)

# Create Python bindings to fast-img-processing
nanobind_add_module(fast_image_processing bindings.cpp)
target_link_libraries(fast_image_processing PRIVATE image_lib)

# Test configuration
if(BUILD_TESTS)
    # Check if in virtual environment
    if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
        set(_pip_args)
    else()
        set(_pip_args "--user")
    endif()

    # Check if pytest is already installed
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pytest"
        RESULT_VARIABLE PYTEST_NOT_FOUND
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(PYTEST_NOT_FOUND)
        message(STATUS "Installing pytest for Python executable: ${Python_EXECUTABLE}")
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pip install ${_pip_args} pytest
            RESULT_VARIABLE PIP_RESULT
            OUTPUT_VARIABLE PIP_OUTPUT
            ERROR_VARIABLE PIP_ERROR
        )

        if(NOT PIP_RESULT EQUAL 0)
            message(WARNING "Failed to install pytest: ${PIP_ERROR}")
        else()
            message(STATUS "Successfully installed pytest")
        endif()
    else()
        message(STATUS "pytest already installed for Python executable: ${Python_EXECUTABLE}")
    endif()

    #Call CTest enable_testing
    enable_testing()
    find_package(GTest REQUIRED)
    include(GoogleTest)
    add_subdirectory(tests)

endif()

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