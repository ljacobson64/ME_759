#include "cuda.h"
#include "math.h"
#include "stdio.h"

#define BLOCK_SIZE 1024

__global__ void reductionDevice(double* d_in, double* d_out, int N) {
  // Setup shared memory
  extern __shared__ float s_data[];

  // Load global memory into shared memory
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[threadIdx.x] = d_in[i];

  // Make sure all the memory in a block is loaded before continuing
  __syncthreads();

  // Add the first and second halves of the array and place the result in the
  // first half. Then add the first and second halves of the original first
  // half, and repeat until the final block sum is computed. The total number of
  // loops is equal to log_2(blockDim.x).
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      s_data[threadIdx.x] += s_data[threadIdx.x + offset];
    __syncthreads();
  }

  // Write the result for each block into d_out
  if (threadIdx.x == 0) d_out[blockIdx.x] = s_data[0];
}

void reductionHost(double* h_in, double* h_ref, int N) {
  double result = 0.f;
  for (int i = 0; i < N; i++) result += h_in[i];
  *h_ref = result;
}

int main(int argc, char* argv[]) {
  int N, M;
  if (argc == 3) {
    N = atoi(argv[1]);
    M = atoi(argv[2]);
  } else {
    printf("Usage: ./p2 <M> <N>\n");
    return 0;
  }

  // For N = 50,000,000 and BLOCK_SIZE = 1024
  //   sizes[0] = 50,000,000
  //   sizes[1] =     48,829
  //   sizes[2] =         48
  //   sizes[3] =          1
  int sizes[4];
  sizes[0] = N;
  sizes[1] = (sizes[0] + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sizes[2] = (sizes[1] + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sizes[3] = (sizes[2] + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Allocate host arrays
  double *h_in, *h_out, *h_ref;
  cudaMallocHost(&h_in, sizeof(double) * N);
  cudaMallocHost(&h_out, sizeof(double));
  h_ref = (double*)malloc(sizeof(double) * N);

  // Allocate device arrays
  double *d_0, *d_1, *d_2, *d_3;
  cudaMalloc(&d_0, sizeof(double) * sizes[0]);
  cudaMalloc(&d_1, sizeof(double) * sizes[1]);
  cudaMalloc(&d_2, sizeof(double) * sizes[2]);
  cudaMalloc(&d_3, sizeof(double) * sizes[3]);

  // Fill host array with random numbers
  srand(73);
  for (int i = 0; i < N; i++)
    h_in[i] = ((double)rand() / RAND_MAX - 0.5f) * 2 * M;

  // Copy host array to device
  cudaMemcpy(d_0, h_in, N * sizeof(double), cudaMemcpyHostToDevice);

  // Perform reduction on device
  reductionDevice <<<sizes[1], BLOCK_SIZE, sizeof(double) * BLOCK_SIZE>>>
      (d_0, d_1, sizes[0]);
  reductionDevice <<<sizes[2], BLOCK_SIZE, sizeof(double) * BLOCK_SIZE>>>
      (d_1, d_2, sizes[1]);
  reductionDevice <<<sizes[3], BLOCK_SIZE, sizeof(double) * BLOCK_SIZE>>>
      (d_2, d_3, sizes[2]);

  // Copy device array back to host
  cudaMemcpy(h_out, d_3, sizeof(double), cudaMemcpyDeviceToHost);

  // Perform reduction on host
  reductionHost(h_in, h_ref, N);

  // Free arrays
  cudaFree(h_in);
  cudaFree(h_out);
  cudaFree(d_0);
  cudaFree(d_1);
  cudaFree(d_2);
  cudaFree(d_3);
  free(h_ref);

  return 0;
}
