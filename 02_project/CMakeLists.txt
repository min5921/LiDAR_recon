cmake_minimum_required(VERSION 3.18)

project(centerpoint_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(centerpoint_voxel_dump
    src/main.cpp
    src/io/bin_point_reader.cpp
    src/io/debug_dump.cpp
    src/voxelization.cpp
)

target_include_directories(centerpoint_voxel_dump
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
    target_compile_options(centerpoint_voxel_dump PRIVATE /W4)
else()
    target_compile_options(centerpoint_voxel_dump PRIVATE -Wall -Wextra -Wpedantic)
endif()

