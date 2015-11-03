#include "cuda.h"
#include "math.h"
#include "stdio.h"

#define BLOCK_SIZE 512

__global__ void reductionDevice(double* d_in, double* d_out, int N) {
  // Setup shared memory
  extern __shared__ float s_data[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load global memory into shared memory
  if (i < N)
    s_data[threadIdx.x] = d_in[i];
  else
    s_data[threadIdx.x] = 0.f;

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

bool checkResults(double* h_out, double* h_ref, double eps) {
  double delta = abs(*h_out - *h_ref);
  if (delta > eps) return false;
  return true;
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
  float dur_max = 1e-30;

  // Setup timing
  int nruns_gpu = 0;
  int nruns_cpu = 0;
  float dur_ex, dur_in, dur_cpu;
  float dur_ex_total = 0.f;
  float dur_in_total = 0.f;
  float dur_cpu_total = 0.f;

  // For N = 50,000,000 and BLOCK_SIZE = 512:
  //   sizes[0] = 50,000,000
  //   sizes[1] =     97,657
  //   sizes[2] =        191
  //   sizes[3] =          1
  int sizes[4];
  sizes[0] = N;
  sizes[1] = (sizes[0] + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sizes[2] = (sizes[1] + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sizes[3] = (sizes[2] + BLOCK_SIZE - 1) / BLOCK_SIZE;

  int shared_size = sizeof(double) * BLOCK_SIZE;

  // Allocate host arrays
  double* h_in, *h_out, *h_ref;
  cudaMallocHost(&h_in, sizeof(double) * N);
  cudaMallocHost(&h_out, sizeof(double));
  cudaMallocHost(&h_ref, sizeof(double));

  // Allocate device arrays
  double* d_0, *d_1, *d_2, *d_3;
  cudaMalloc(&d_0, sizeof(double) * sizes[0]);
  cudaMalloc(&d_1, sizeof(double) * sizes[1]);
  cudaMalloc(&d_2, sizeof(double) * sizes[2]);
  cudaMalloc(&d_3, sizeof(double) * sizes[3]);

  // Fill host array with random numbers
  srand(73);
  for (int i = 0; i < N; i++)
    h_in[i] = ((double)rand() / RAND_MAX - 0.5f) * 2 * M;

  while (dur_in_total < dur_max) {
    nruns_gpu++;

    // Setup timing
    cudaEvent_t start_ex, end_ex, start_in, end_in;
    cudaEventCreate(&start_ex);
    cudaEventCreate(&end_ex);
    cudaEventCreate(&start_in);
    cudaEventCreate(&end_in);

    // Copy host array to device
    cudaEventRecord(start_in, 0);
    cudaMemcpy(d_0, h_in, N * sizeof(double), cudaMemcpyHostToDevice);

    // Perform reduction on device
    cudaEventRecord(start_ex, 0);
    reductionDevice <<<sizes[1], BLOCK_SIZE, shared_size>>>
        (d_0, d_1, sizes[0]);
    reductionDevice <<<sizes[2], BLOCK_SIZE, shared_size>>>
        (d_1, d_2, sizes[1]);
    reductionDevice <<<sizes[3], BLOCK_SIZE, shared_size>>>
        (d_2, d_3, sizes[2]);
    cudaEventRecord(end_ex, 0);
    cudaEventSynchronize(end_ex);

    // Copy device array back to host
    cudaMemcpy(h_out, d_3, sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(end_in, 0);
    cudaEventSynchronize(end_in);

    // Calculate durations
    cudaEventElapsedTime(&dur_ex, start_ex, end_ex);
    cudaEventElapsedTime(&dur_in, start_in, end_in);
    dur_ex_total += dur_ex;
    dur_in_total += dur_in;
  }

  while (dur_cpu_total < dur_max) {
    nruns_cpu++;

    // Setup timing
    cudaEvent_t start_cpu, end_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&end_cpu);

    // Perform reduction on host
    cudaEventRecord(start_cpu, 0);
    reductionHost(h_in, h_ref, N);
    cudaEventRecord(end_cpu, 0);
    cudaEventSynchronize(end_cpu);

    // Calculate durations
    cudaEventElapsedTime(&dur_cpu, start_cpu, end_cpu);
    dur_cpu_total += dur_cpu;
  }

  dur_ex = dur_ex_total / nruns_gpu;
  dur_in = dur_in_total / nruns_gpu;
  dur_cpu = dur_cpu_total / nruns_cpu;

  // Compare device and host results
  double eps = (double)M * 2 * 0.001f;
  bool testPassed = checkResults(h_out, h_ref, eps);
  if (testPassed)
    printf("Test PASSED\n");
  else
    printf("Test FAILED\n");

  // Print stuff
  printf("GPU result: %20.14f\n", *h_out);
  printf("CPU result: %20.14f\n", *h_ref);
  printf("Num runs GPU: %10d\n", nruns_gpu);
  printf("Num runs CPU: %10d\n", nruns_cpu);
  printf("GPU execution time (exclusive): %12.6f\n", dur_ex);
  printf("GPU execution time (inclusive): %12.6f\n", dur_in);
  printf("CPU execution time:             %12.6f\n", dur_cpu);
  printf("\n");

  // Free arrays
  cudaFree(h_in);
  cudaFree(h_out);
  cudaFree(h_ref);
  cudaFree(d_0);
  cudaFree(d_1);
  cudaFree(d_2);
  cudaFree(d_3);

  return 0;
}
