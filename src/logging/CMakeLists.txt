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

# Platform-specific compiler options
if(MSVC)
    target_compile_options(logging 
        PRIVATE 
            /W4 #Warning level 4
            /WX #Treat warning as error
            /wd4251 # Disable class needs DLL-interface warning
            /MP     # Enable multi-processor compilation
    )
else()
    target_compile_options(logging 
        PRIVATE 
            -Wall 
            -Wextra 
            -Wpedantic
            -Werror
    )
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

# macOS specific settings
if(APPLE)
    set_target_properties(logging PROPERTIES
        MACOSX_RPATH ON
        INSTALL_RPATH "@loader_path;@loader_path/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()