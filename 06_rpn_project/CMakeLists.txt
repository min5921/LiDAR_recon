cmake_minimum_required(VERSION 3.24)

project(centerpoint_rpn_cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(centerpoint_rpn_cuda_demo
    src/main.cpp
    src/io/rpn_dump_writer.cpp
    cuda/rpn_kernels.cu
)

add_executable(centerpoint_rpn_full_cuda
    src/main_full.cpp
    src/io/bev_dump_reader.cpp
    src/io/rpn_weight_reader.cpp
    src/io/rpn_output_writer.cpp
    cuda/rpn_full_cuda.cu
)

target_include_directories(centerpoint_rpn_cuda_demo
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(centerpoint_rpn_full_cuda
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(centerpoint_rpn_full_cuda PRIVATE CUDA::cublas)

set_target_properties(centerpoint_rpn_cuda_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(centerpoint_rpn_full_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if (MSVC)
    target_compile_options(centerpoint_rpn_cuda_demo PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
    target_compile_options(centerpoint_rpn_full_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(centerpoint_rpn_cuda_demo PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    target_compile_options(centerpoint_rpn_full_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()
