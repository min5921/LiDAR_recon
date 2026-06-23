cmake_minimum_required(VERSION 3.18)

project(centerpoint_pillar_feature LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(centerpoint_decorate_pillars
    src/main.cpp
    src/io/decorated_dump_writer.cpp
    src/io/voxel_dump_reader.cpp
    src/pillar_feature.cpp
)

target_include_directories(centerpoint_decorate_pillars
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
    target_compile_options(centerpoint_decorate_pillars PRIVATE /W4)
else()
    target_compile_options(centerpoint_decorate_pillars PRIVATE -Wall -Wextra -Wpedantic)
endif()

