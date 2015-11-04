#include "cuda.h"
#include "limits.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#define BLOCK_SIZE 512
#define DOUBLE_BLOCK 1024
#define MAX_GRID_DIM 65535

__global__ void reductionDevice(double* d_in, double* d_out, int N) {
  extern __shared__ double s_data[];

  // Indexing
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned int i = blockId * (blockDim.x * 2) + threadIdx.x;

  // Each thread loads 2 values into shared memory and adds them
  if (i + blockDim.x < N)
    s_data[threadIdx.x] = d_in[i] + d_in[i + blockDim.x];
  else if (i < N)
    s_data[threadIdx.x] = d_in[i];
  else
    s_data[threadIdx.x] = 0.f;

  // Make sure all the memory in the block is loaded before continuing
  __syncthreads();

  // Add the first and second halves of the array and place the result in the
  // first half. Then add the first and second halves of the original first
  // half, and repeat until the entire block is reduced.
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      s_data[threadIdx.x] += s_data[threadIdx.x + offset];
    __syncthreads();
  }

  // Write the result for each block into d_out
  if (threadIdx.x == 0 && i < N) d_out[blockId] = s_data[0];
}

void exitUsage() {
  printf("Usage: ./p2 [<M> <N> [<dur_max>]]\n");
  exit(EXIT_SUCCESS);
}

void parseInput(int argc, char* argv[], unsigned int &N, unsigned int &M,
                float &dur_max) {
  if (argc == 1) {
    N = 50000000;
    M = 5;
    dur_max = 1000.f;
    return;
  }
  if (argc != 3 && argc != 4) exitUsage();
  if (sscanf(argv[1], "%u", &N) != 1) exitUsage();
  if (sscanf(argv[2], "%u", &M) != 1) exitUsage();
  if (argc == 3) {
    dur_max = 1000.f;
    return;
  }
  if (sscanf(argv[3], "%f", &dur_max) != 1) exitUsage();
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
  unsigned int N, M;
  float dur_max;
  parseInput(argc, argv, N, M, dur_max);

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

  // Calculate the tree depth
  int tree_depth = 0;
  int length = N;
  while (length > 1) {
    length = (length + DOUBLE_BLOCK - 1) / DOUBLE_BLOCK;
    tree_depth++;
  }

  // Calculate the lengths of the device arrays
  int lengths[tree_depth + 1];
  lengths[0] = N;
  for (int i = 1; i < tree_depth + 1; i++)
    lengths[i] = (lengths[i - 1] + DOUBLE_BLOCK - 1) / DOUBLE_BLOCK;

  // Setup grid
  dim3 dimGrid[tree_depth];
  for (int i = 0; i < tree_depth; i++) {
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
  double* d_arr[tree_depth + 1];
  for (int i = 0; i < tree_depth + 1; i++)
    d_arr[i] = AllocateDeviceArray(sizeof(double) * lengths[i]);

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
    cudaMemcpy(d_arr[0], h_in, N * sizeof(double), cudaMemcpyHostToDevice);

    // Perform reduction on device
    cudaEventRecord(start_ex, 0);
    for (int i = 0; i < tree_depth; i++)
      reductionDevice <<<dimGrid[i], BLOCK_SIZE, shared_size>>>
          (d_arr[i], d_arr[i + 1], lengths[i]);
    cudaEventRecord(end_ex, 0);
    cudaEventSynchronize(end_ex);

    // Copy device array back to host
    cudaMemcpy(h_out, d_arr[tree_depth], sizeof(double), cudaMemcpyDeviceToHost);
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
  printf("N: %u\n", N);
  printf("M: %u\n", M);
  printf("Block size: %d\n", BLOCK_SIZE);
  printf("Tree depth: %d\n", tree_depth);
  printf("gridDims: ");
  for (int i = 0; i < tree_depth - 1; i++)
    printf("%dx%d, ", dimGrid[i].y, dimGrid[i].x);
  printf("%dx%d\n", dimGrid[tree_depth - 1].y, dimGrid[tree_depth - 1].x);
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
  for (int i = 0; i < tree_depth + 1; i++) cudaFree(d_arr[i]);

  return 0;
}
