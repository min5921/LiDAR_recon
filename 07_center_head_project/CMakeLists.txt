cmake_minimum_required(VERSION 3.24)
project(centerpoint_head_cuda LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
add_executable(centerpoint_head_cuda
    src/main.cpp
    src/head_io.cpp
    cuda/center_head_cuda.cu)
target_include_directories(centerpoint_head_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(centerpoint_head_cuda PRIVATE CUDA::cublas)
set_target_properties(centerpoint_head_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(MSVC)
  target_compile_options(centerpoint_head_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
endif()
