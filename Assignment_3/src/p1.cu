#include <stdio.h>

__global__ void addBlockThread(int* data) {
    int ind = blockDim.x*blockIdx.x + threadIdx.x;
    int sum = blockIdx.x + threadIdx.x;
    data[ind] = sum;
    printf("%6d  %6d  %6d\n", blockIdx.x, threadIdx.x, sum);
}

int main() {
    int num_blocks = 2;
    int num_threads = 8;
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_threads);
    int num_ints = num_blocks*num_threads;
    
    int hostArray[num_ints];
    int *devArray;
    
    // Allocate memory on the device. devArray is a pointer to the allocated
    // memory.
    cudaMalloc((void**)&devArray, sizeof(int)*num_ints);
    
    // Invoke the device kernel which adds the block and thread indices
    printf("\nValues written to the device array:\n");
    printf("%6s  %6s  %6s\n", "Block", "Thread", "Sum");
    addBlockThread<<<dimGrid, dimBlock>>>(devArray);
    
    // Bring the results pointed to by devArray back to hostArray
    cudaMemcpy(&hostArray, devArray, sizeof(int)*num_ints,
               cudaMemcpyDeviceToHost);
    
    // Print the results
    printf("\nValues stored in the host array:\n");
    for (int i = 0; i < num_ints; i++) { printf("%d ", hostArray[i]); }
    printf("\n");
    
    // Free the device memory
    cudaFree(devArray);
    
    return 0;
}
