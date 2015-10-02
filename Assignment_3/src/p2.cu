#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void sumArrays(double* dA, double* dB, double* dC) {
    int ind = blockDim.x*blockIdx.x + threadIdx.x;
    dC[ind] = dA[ind] + dB[ind];
}

int int_power(int base, int exponent) {
    int result = 1;
    for (int i = 0; i < exponent; i++) { result *= base; }
    return result;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {  exit(1); }
    int exponent = atoi(argv[1]);
    int nthreads = atoi(argv[2]);
    
    int N = int_power(2, exponent);  // Number of random numbers
    int nblocks = N/nthreads;
    
    // Allocate host arrays
    int bytes = sizeof(double)*N;
    double *hA, *hB, *hC, *refC, *difC;
    hA   = (double*)malloc(bytes);
    hB   = (double*)malloc(bytes);
    hC   = (double*)malloc(bytes);
    refC = (double*)malloc(bytes);
    difC = (double*)malloc(bytes);
    
    // Allocate device arrays
    double *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    
    // Fill host arrays with random numbers between -10 and 10 and sum them for
    // reference
    srand(1443740650);
    for (int i = 0; i < N; i++) {
        hA[i] = ((double)rand()/(double)RAND_MAX - 0.5)*20;
        hB[i] = ((double)rand()/(double)RAND_MAX - 0.5)*20;
        refC[i] = hA[i] + hB[i];
    }
    
    // Copy host arrays to the device
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);
    
    // Set up timing
    float duration_ex;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    // Invoke the device kernel which sums the arrays
    cudaEventRecord(start, 0);
    
    sumArrays<<<nblocks, nthreads>>>(dA, dB, dC);
    
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&duration, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    
    // Copy the sum array back to the host
    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    
    // Calculate the difference between the sum arrays and find the maximum
    // absolute difference
    double max_dif = 0.0;
    for (int i = 0; i < N; i++) {
        difC[i] = hC[i] - refC[i];
        if (abs(difC[i]) > max_dif) { max_dif = abs(difC[i]); }
    }
    
    // Free memory
    free(hA);
    free(hB);
    free(hC);
    free(refC);
    free(difC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    // Print some information
    printf("Number of integers:    %12d\n", N);
    printf("Maximum difference:    %12.4e\n", max_dif);
    printf("Time taken:            %12.6e ms\n", duration_ex);
    
    return 0;
}
