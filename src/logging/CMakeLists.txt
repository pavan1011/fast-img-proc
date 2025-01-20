add_library(logging
    logging.cpp
)

target_include_directories(logging
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
)

target_compile_features(logging 
    PUBLIC 
        cxx_std_20
)

if(MSVC)
    target_compile_options(logging 
        PRIVATE /W4 /WX)
else()
    target_compile_options(logging 
        PRIVATE 
            -Wall 
            -Wextra 
            -Wpedantic
            -Werror)
endif()

find_package(Threads REQUIRED)
target_link_libraries(logging
    PUBLIC
        Threads::Threads
        project_options
)

# Set properties
set_target_properties(logging PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)