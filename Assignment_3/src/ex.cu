// #include <cutil_inline.h>
#include <iostream>

__global__ void simpleKernel(int* data) {
    // Write something trivial to the global memory
    data[threadIdx.x] = blockIdx.x + threadIdx.x;
}

int main() {
    int hostArray[4], *devArray;
    
    // Allocate memory on the device (GPU)
    cudaMalloc((void**) &devArray, sizeof(int)*4);
    
    // Invoke GPU kernel with 1 block that has 4 threads
    simpleKernel<<<1,4>>>(devArray);
    
    // Bring the result back from the GPU into the hostArray
    cudaMemcpy(&hostArray, devArray, sizeof(int)*4, cudaMemcpyDeviceToHost);
    
    // Print out the results to confirm that things are looking good
    std::cout << "Values stored in hostArray: ";
    std::cout << hostArray[0] << ", ";
    std::cout << hostArray[1] << ", ";
    std::cout << hostArray[2] << ", ";
    std::cout << hostArray[3] << std::endl;
    
    // Release the memory allocated on the GPU
    cudaFree(devArray);
    
    return 0;
}
