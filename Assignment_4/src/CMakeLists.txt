cmake_minimum_required (VERSION 2.8)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_20,code=sm_20)

cuda_add_executable(p2 p2.cu)
cuda_add_executable(p3 matrixadd.cu)
