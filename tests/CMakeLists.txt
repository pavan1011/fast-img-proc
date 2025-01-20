# tests/CMakeLists.txt

# Create test executable
add_executable(cpp_tests
    cpp/test_image.cpp
    cpp/test_cpu_processing.cpp
)

# Add GPU tests if CUDA is enabled
if(USE_CUDA)
    target_sources(cpp_tests
        PRIVATE
            cpp/test_gpu_processing.cpp
    )
    set_source_files_properties(
        cpp/test_gpu_processing.cpp
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
    )
    set_target_properties(cpp_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# Link with GTest and project libraries
target_link_libraries(cpp_tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        image_lib
        project_options
        logging
)

# Add include directories
target_include_directories(cpp_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Discover tests
gtest_discover_tests(cpp_tests)

# Python tests configuration
if(Python_FOUND)
    # Copy Python test files to build directory
    file(COPY 
    ${CMAKE_SOURCE_DIR}/tests/python/test_processing.py
    ${CMAKE_SOURCE_DIR}/tests/python/images/
    DESTINATION ${CMAKE_BINARY_DIR}/tests/python)
    
    # Add custom target for Python tests
    add_custom_target(python_tests
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_BINARY_DIR}/tests/python
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS fast_image_processing
        COMMENT "Running Python tests"
    )
    
    # Add to test suite
    add_test(NAME python_tests 
             COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_BINARY_DIR}/tests/python
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()