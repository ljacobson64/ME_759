cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_20,code=sm_20 -Xptxas=-v)
set(CMAKE_BUILD_TYPE "Release")

cuda_add_executable(p1 stencil_1d.cu)
cuda_add_executable(p2 2Dconvolution.cu)
