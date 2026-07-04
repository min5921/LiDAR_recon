cmake_minimum_required(VERSION 3.24)

project(centerpoint_decode LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(centerpoint_decode
    src/main.cpp
    src/decode_io.cpp
    cuda/decode_cuda.cu
)

target_include_directories(centerpoint_decode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(centerpoint_decode PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(MSVC)
    target_compile_options(centerpoint_decode PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
endif()
