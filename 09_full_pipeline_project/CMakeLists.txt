cmake_minimum_required(VERSION 3.18)

project(centerpoint_full_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(waymo_frame_inspect
    src/main.cpp
    src/waymo_sensor_archive.cpp
)

target_include_directories(waymo_frame_inspect
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
    target_compile_options(waymo_frame_inspect PRIVATE /W4)
else()
    target_compile_options(waymo_frame_inspect PRIVATE -Wall -Wextra -Wpedantic)
endif()

