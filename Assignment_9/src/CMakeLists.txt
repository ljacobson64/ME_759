cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_20,code=sm_20 -Xptxas=-v)
set(CMAKE_BUILD_TYPE "Release")

cuda_add_executable(p1 p1.cu)
