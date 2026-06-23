cmake_minimum_required(VERSION 3.18)

project(centerpoint_pfn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(centerpoint_pfn_dummy
    src/main.cpp
    src/io/decorated_dump_reader.cpp
    src/io/pillar_feature_writer.cpp
    src/pfn.cpp
)

target_include_directories(centerpoint_pfn_dummy
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
    target_compile_options(centerpoint_pfn_dummy PRIVATE /W4)
else()
    target_compile_options(centerpoint_pfn_dummy PRIVATE -Wall -Wextra -Wpedantic)
endif()

