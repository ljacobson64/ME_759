#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16

__global__ void cudaMultiplyArrays(int* dA, int* dB, int* dC,
                                   int hA, int wA, int wB, int wC) {
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  if (y > hA || x > wB) return;
  int result = 0;
  for (int e = 0; e < wA; e++)
    result += dA[y*wA + e]*dB[e*wB + x];
  dC[y*wC + x] = result;
}

void fill_array(int* A, int hA, int wA) {
  for (int i = 0; i < hA; i++)
    for (int j = 0; j < wA; j++)
      A[i*wA + j] = i + j;
}

int main() {
  // Array sizes
  int m = 16;
  int n = 32;
  int p =  1;
  int hA = m, wA = n;
  int hB = n, wB = p;
  int hC = m, wC = p;
  int sA = hA*wA;
  int sB = hB*wB;
  int sC = hC*wC;

  // Allocate host arrays
  int *A, *B, *C;
  A = (int*)malloc(sizeof(int)*sA);
  B = (int*)malloc(sizeof(int)*sB);
  C = (int*)malloc(sizeof(int)*sC);

  // Allocate device arrays
  int *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(int)*sA);
  cudaMalloc(&dB, sizeof(int)*sB);
  cudaMalloc(&dC, sizeof(int)*sC);

  // Fill A and B with some integers
  fill_array(A, hA, wA);
  fill_array(B, hB, wB);

  // Set up block grid
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((wC + dimBlock.x - 1)/dimBlock.x,
               (hC + dimBlock.y - 1)/dimBlock.y);

  // Set up timing
  struct timespec start_in, end_in;
  int num_runs = 65536;
  long dur_in_ns;
  double dur_in = 0.0, dur_in_total = 0.0;
  double dur_in_max = 0.0, dur_in_min = 1e99;

  for (int i = 0; i < num_runs; i++) {
    // Start inclusive timing
    clock_gettime(CLOCK_MONOTONIC, &start_in);

    // Copy host arrays to the device
    cudaMemcpy(dA, A, sizeof(int)*sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(int)*sB, cudaMemcpyHostToDevice);

    // Invoke the device kernel which multiplies the arrays
    cudaMultiplyArrays<<<dimGrid, dimBlock>>>(dA, dB, dC, hA, wA, wB, wC);

    // Copy the result array back to the host
    cudaMemcpy(C, dC, sizeof(int)*sC, cudaMemcpyDeviceToHost);

    // End inclusive timing
    clock_gettime(CLOCK_MONOTONIC, &end_in);

    // Calculate duration
    dur_in_ns = (end_in.tv_sec - start_in.tv_sec)*1000000000l +
                 end_in.tv_nsec - start_in.tv_nsec;
    dur_in = (double)(dur_in_ns/1000000.0);
    dur_in_total += dur_in;
    if (dur_in > dur_in_max) dur_in_max = dur_in;
    if (dur_in < dur_in_min) dur_in_min = dur_in;
  }

  // Write result to file
  FILE *fp;
  fp = fopen("problem2.out", "w");
  for (int i = 0; i < hC; i++)
    for (int j = 0; j < wC; j++)
      fprintf(fp, "%12d\n", C[i*wC + j]);
  fclose(fp);

  // Free memory
  free(A);
  free(B);
  free(C);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  // Get device properties
  cudaDeviceProp gpu_props;
  cudaGetDeviceProperties(&gpu_props, 0);

  // Print some information
  printf("Device name: %s\n", gpu_props.name);
  printf("Number of runs: %12d\n", num_runs);
  printf("Inclusive time (maximum): %12.6f ms\n", dur_in_max);
  printf("Inclusive time (average): %12.6f ms\n", dur_in_total/num_runs);
  printf("Inclusive time (minimum): %12.6f ms\n", dur_in_min);

  return 0;
}
