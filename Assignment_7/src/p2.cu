#include "cuda.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#define TREE_DEPTH 3
#define BLOCK_SIZE 512
#define MAX_GRID_DIM 65535

__global__ void reductionDevice(double* d_in, double* d_out, int N) {
  extern __shared__ double s_data[];
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int i = blockId * blockDim.x + threadIdx.x;

  // Load data into shared memory
  if (i < N)
    s_data[threadIdx.x] = d_in[i];
  else
    s_data[threadIdx.x] = 0.f;

  // Make sure all the memory in the block is loaded before continuing
  __syncthreads();

  // Add the first and second halves of the array and place the result in the
  // first half. Then add the first and second halves of the original first
  // half, and repeat until the entire block is reduced.
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      s_data[threadIdx.x] += s_data[threadIdx.x + offset];
    __syncthreads();
  }

  // Write the result for each block into d_out
  if (threadIdx.x == 0 && i < N) d_out[blockId] = s_data[0];
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

double* AllocateHostArray(int size) {
  double* h_array;
  cudaError_t code = cudaMallocHost(&h_array, size);
  if (code != cudaSuccess) {
    printf("Memory allocation on the host was unsuccessful.\n");
    exit(EXIT_FAILURE);
  }
  return h_array;
}

double* AllocateDeviceArray(int size) {
  double* d_array;
  cudaError_t code = cudaMalloc(&d_array, size);
  if (code != cudaSuccess) {
    printf("Memory allocation on the device was unsuccessful.\n");
    exit(EXIT_FAILURE);
  }
  return d_array;
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
  float dur_max = 1000.f;

  // Setup timing
  int nruns_gpu = 0;
  int nruns_cpu = 0;
  float dur_ex, dur_in, dur_cpu;
  float dur_ex_total = 0.f;
  float dur_in_total = 0.f;
  float dur_cpu_total = 0.f;
  float dur_ex_min = 1e99;
  float dur_in_min = 1e99;
  float dur_cpu_min = 1e99;

  // Setup grid
  int lengths[TREE_DEPTH + 1];
  lengths[0] = N;
  for (int i = 1; i < TREE_DEPTH + 1; i++)
    lengths[i] = (lengths[i - 1] + BLOCK_SIZE - 1) / BLOCK_SIZE;

  dim3 dimGrid[3];
  for (int i = 0; i < TREE_DEPTH; i++) {
    if (lengths[i + 1] > MAX_GRID_DIM) {
      dimGrid[i].x = MAX_GRID_DIM;
      dimGrid[i].y = (lengths[i + 1] + MAX_GRID_DIM - 1) / MAX_GRID_DIM;
    } else {
      dimGrid[i].x = lengths[i + 1];
      dimGrid[i].y = 1;
    }
  }

  int shared_size = sizeof(double) * BLOCK_SIZE;

  // Allocate host arrays
  double* h_in = AllocateHostArray(sizeof(double) * N);
  double* h_out = AllocateHostArray(sizeof(double));
  double* h_ref = AllocateHostArray(sizeof(double));

  // Allocate device arrays
  double* d_0 = AllocateDeviceArray(sizeof(double) * lengths[0]);
  double* d_1 = AllocateDeviceArray(sizeof(double) * lengths[1]);
  double* d_2 = AllocateDeviceArray(sizeof(double) * lengths[2]);
  double* d_3 = AllocateDeviceArray(sizeof(double) * lengths[3]);

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
    reductionDevice <<<dimGrid[0], BLOCK_SIZE, shared_size>>>
        (d_0, d_1, lengths[0]);
    reductionDevice <<<dimGrid[1], BLOCK_SIZE, shared_size>>>
        (d_1, d_2, lengths[1]);
    reductionDevice <<<dimGrid[2], BLOCK_SIZE, shared_size>>>
        (d_2, d_3, lengths[2]);
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
    if (dur_ex < dur_ex_min) dur_ex_min = dur_ex;
    if (dur_in < dur_in_min) dur_in_min = dur_in;
    if (dur_in_total == 0.f) break;
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
    if (dur_cpu < dur_cpu_min) dur_cpu_min = dur_cpu;
    if (dur_cpu_total == 0.f) break;
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
  printf("N: %d\n", N);
  printf("M: %d\n", M);
  printf("Block size: %d\n", BLOCK_SIZE);
  printf("gridDims: %dx%d, %dx%d, %dx%d\n", dimGrid[0].y, dimGrid[0].x,
         dimGrid[1].y, dimGrid[1].x, dimGrid[2].y, dimGrid[2].x);
  printf("GPU result: %24.14f\n", *h_out);
  printf("CPU result: %24.14f\n", *h_ref);
  printf("Timing results %12s %12s %8s\n", "Average", "Minimum", "Num_runs");
  printf("GPU exclusive: %12.6f %12.6f %8d\n", dur_ex, dur_ex_min, nruns_gpu);
  printf("GPU inclusive: %12.6f %12.6f %8d\n", dur_in, dur_in_min, nruns_gpu);
  printf("CPU:           %12.6f %12.6f %8d\n", dur_cpu, dur_cpu_min, nruns_cpu);
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
