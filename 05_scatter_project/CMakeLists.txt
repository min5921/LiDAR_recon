cmake_minimum_required(VERSION 3.18)
project(centerpoint_scatter LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_executable(centerpoint_scatter
    src/main.cpp
    src/scatter.cpp
    src/io/scatter_input_reader.cpp
    src/io/bev_feature_writer.cpp
)
target_include_directories(centerpoint_scatter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (MSVC)
    target_compile_options(centerpoint_scatter PRIVATE /W4)
else()
    target_compile_options(centerpoint_scatter PRIVATE -Wall -Wextra -Wpedantic)
endif()
