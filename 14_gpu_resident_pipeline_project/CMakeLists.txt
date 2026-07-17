cmake_minimum_required(VERSION 3.24)

project(centerpoint_gpu_resident_pipeline LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(centerpoint_gpu_preprocess
    src/main.cpp
    src/io.cpp
    src/pfn_weights.cpp
    cuda/gpu_preprocess.cu
)

add_executable(centerpoint_gpu_rpn
    src/main_rpn.cpp
    src/io.cpp
    src/pfn_weights.cpp
    src/rpn_weights.cpp
    cuda/gpu_preprocess.cu
    cuda/gpu_rpn.cu
)

add_executable(centerpoint_gpu_full
    src/main_full.cpp
    src/io.cpp
    src/pfn_weights.cpp
    src/rpn_weights.cpp
    src/head_weights.cpp
    cuda/gpu_preprocess.cu
    cuda/gpu_rpn.cu
    cuda/gpu_center_head.cu
    cuda/gpu_postprocess.cu
)

target_include_directories(centerpoint_gpu_preprocess
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(centerpoint_gpu_rpn
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(centerpoint_gpu_full
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(centerpoint_gpu_rpn PRIVATE CUDA::cublas)
target_link_libraries(centerpoint_gpu_full PRIVATE CUDA::cublas)

target_compile_definitions(centerpoint_gpu_preprocess PRIVATE
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
)

target_compile_definitions(centerpoint_gpu_rpn PRIVATE
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
)

target_compile_definitions(centerpoint_gpu_full PRIVATE
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
)

set_target_properties(centerpoint_gpu_preprocess PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(centerpoint_gpu_rpn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(centerpoint_gpu_full PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(centerpoint_gpu_preprocess PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
)

target_compile_options(centerpoint_gpu_rpn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
)

target_compile_options(centerpoint_gpu_full PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
)

if (MSVC)
    target_compile_options(centerpoint_gpu_preprocess PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
    target_compile_options(centerpoint_gpu_rpn PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
    target_compile_options(centerpoint_gpu_full PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(centerpoint_gpu_preprocess PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    target_compile_options(centerpoint_gpu_rpn PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
    target_compile_options(centerpoint_gpu_full PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()
