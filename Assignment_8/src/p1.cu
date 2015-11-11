#include "cuda.h"
#include "limits.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

// Definitions
#define MAX_RAND 2
#define DEFAULT_NUM_ELEMENTS 16777216
#define BLOCK_SIZE 512
#define DOUBLE_BLOCK 1024
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(x) ((x) >> LOG_NUM_BANKS)

__global__ void prescanKernel(float* d_in, float* d_out, float* d_sums,
                              unsigned int N) {
  extern __shared__ float s_data[];
  unsigned int shared_end =
      DOUBLE_BLOCK + CONFLICT_FREE_OFFSET(DOUBLE_BLOCK) - 2;

  // Indexing
  unsigned int offset = 1;
  unsigned int ai = threadIdx.x;
  unsigned int bi = threadIdx.x + BLOCK_SIZE;
  unsigned int ag = ai + DOUBLE_BLOCK * blockIdx.x;
  unsigned int bg = bi + DOUBLE_BLOCK * blockIdx.x;
  unsigned int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  unsigned int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  // Load data into shared memory
  s_data[ai + bankOffsetA] = (ag < N) ? d_in[ag] : 0.f;
  s_data[bi + bankOffsetB] = (bg < N) ? d_in[bg] : 0.f;

  // Build sum in place up the tree
  for (unsigned int d = BLOCK_SIZE; d > 0; d >>= 1) {
    __syncthreads();
    if (threadIdx.x < d) {
      unsigned int ai = offset * (2 * threadIdx.x + 1) - 1;
      unsigned int bi = offset * (2 * threadIdx.x + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      s_data[bi] += s_data[ai];
    }
    offset <<= 1;
  }

  // Write the last element of shared memory to the auxilary array and clear it
  if (threadIdx.x == 0) {
    d_sums[blockIdx.x] = s_data[shared_end];
    s_data[shared_end] = 0.f;
  }

  // Traverse down the tree and build scan
  for (unsigned int d = 1; d <= BLOCK_SIZE; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (threadIdx.x < d) {
      unsigned int ai = offset * (2 * threadIdx.x + 1) - 1;
      unsigned int bi = offset * (2 * threadIdx.x + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      float temp = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += temp;
    }
  }

  // Write results to global memory
  __syncthreads();
  if (ag < N) d_out[ag] = s_data[ai + bankOffsetA];
  if (bg < N) d_out[bg] = s_data[bi + bankOffsetB];
}

__global__ void additionKernel(float* d_in, float* d_out, float* d_sums,
                               unsigned int N) {
  unsigned int i = DOUBLE_BLOCK * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  d_out[i] = d_in[i] + d_sums[blockIdx.x];
  d_out[i + BLOCK_SIZE] = d_in[i + BLOCK_SIZE] + d_sums[blockIdx.x];
}

void prescanOnDevice(float* h_in, float* h_out, float** d_arr, unsigned int N,
                     int tree_depth, unsigned int* lengths, dim3* dimBlock,
                     dim3* dimGrid, unsigned int shared_size, float& dur_ex,
                     float& dur_in) {
  // Setup timing
  cudaEvent_t start_ex, end_ex, start_in, end_in;
  cudaEventCreate(&start_ex);
  cudaEventCreate(&end_ex);
  cudaEventCreate(&start_in);
  cudaEventCreate(&end_in);

  // Copy host array to device
  cudaEventRecord(start_in, 0);
  cudaMemcpy(d_arr[0], h_in, lengths[0] * sizeof(float),
             cudaMemcpyHostToDevice);

  // Perform prescan on device
  cudaEventRecord(start_ex, 0);
  for (int i = 0; i < tree_depth; i++)
    prescanKernel <<<dimGrid[i], dimBlock[i], shared_size>>>
        (d_arr[i], d_arr[i], d_arr[i + 1], lengths[i]);
  for (int i = tree_depth - 2; i >= 0; i--)
    additionKernel <<<dimGrid[i], dimBlock[i]>>>
        (d_arr[i], d_arr[i], d_arr[i + 1], lengths[i]);
  cudaEventRecord(end_ex, 0);
  cudaEventSynchronize(end_ex);

  // Copy device array back to host
  cudaMemcpy(h_out, d_arr[0], lengths[0] * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(end_in, 0);
  cudaEventSynchronize(end_in);

  // Calculate durations
  cudaEventElapsedTime(&dur_ex, start_ex, end_ex);
  cudaEventElapsedTime(&dur_in, start_in, end_in);

  // Cleanup timing
  cudaEventDestroy(start_ex);
  cudaEventDestroy(end_ex);
  cudaEventDestroy(start_in);
  cudaEventDestroy(end_in);
}

void prescanOnHost(float* h_in, float* h_ref, unsigned int N, float& dur_cpu) {
  // Setup timing
  cudaEvent_t start_cpu, end_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&end_cpu);

  // Perform prescan on host
  cudaEventRecord(start_cpu, 0);
  h_ref[0] = 0;
  for (unsigned int i = 1; i < N; i++) h_ref[i] = h_in[i - 1] + h_ref[i - 1];
  cudaEventRecord(end_cpu, 0);
  cudaEventSynchronize(end_cpu);

  // Calculate duration
  cudaEventElapsedTime(&dur_cpu, start_cpu, end_cpu);

  // Cleanup timing
  cudaEventDestroy(start_cpu);
  cudaEventDestroy(end_cpu);
}

unsigned int checkResults(float* h_out, float* h_ref, unsigned int N,
                          float eps) {
  unsigned int nDiffs = 0;
  for (unsigned int i = 0; i < N; i++) {
    float delta = abs(h_out[i] - h_ref[i]);
    if (delta > eps) nDiffs++;
  }
  return nDiffs;
}

float* allocateHostArray(unsigned int size) {
  float* h_array;
  cudaError_t code = cudaMallocHost(&h_array, size);
  if (code != cudaSuccess) {
    printf("Memory allocation on the host was unsuccessful.\n");
    exit(EXIT_FAILURE);
  }
  return h_array;
}

float* allocateDeviceArray(unsigned int size) {
  float* d_arr;
  cudaError_t code = cudaMalloc(&d_arr, size);
  if (code != cudaSuccess) {
    printf("Memory allocation on the device was unsuccessful.\n");
    exit(EXIT_FAILURE);
  }
  return d_arr;
}

void exitUsage() {
  printf("Usage: ./p2 N [dur_max]\n");
  exit(EXIT_SUCCESS);
}

void parseInput(int argc, char** argv, unsigned int& N, float& dur_max) {
  if (argc == 1) {
    N = DEFAULT_NUM_ELEMENTS;
    dur_max = 1000.f;
    return;
  }
  if (argc != 2 && argc != 3) exitUsage();
  if (sscanf(argv[1], "%u", &N) != 1) exitUsage();
  if (argc == 2) {
    dur_max = 1000.f;
    return;
  }
  if (sscanf(argv[2], "%f", &dur_max) != 1) exitUsage();
  dur_max *= 1000;
}

int main(int argc, char** argv) {
  unsigned int N;
  float dur_max;
  parseInput(argc, argv, N, dur_max);

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
  {
    unsigned int length = N;
    while (length > 1) {
      length = (length + DOUBLE_BLOCK - 1) / DOUBLE_BLOCK;
      tree_depth++;
    }
  }

  // Calculate the lengths of the device arrays
  unsigned int lengths[tree_depth + 1];
  lengths[0] = N;
  for (int i = 1; i < tree_depth + 1; i++)
    lengths[i] = (lengths[i - 1] + DOUBLE_BLOCK - 1) / DOUBLE_BLOCK;

  // Setup grid
  dim3 dimBlock[tree_depth];
  dim3 dimGrid[tree_depth];
  for (int i = 0; i < tree_depth; i++) {
    dimBlock[i].x = BLOCK_SIZE;
    dimGrid[i].x = lengths[i + 1];
  }

  // Shared memory size
  unsigned int shared_size =
      (DOUBLE_BLOCK + CONFLICT_FREE_OFFSET(DOUBLE_BLOCK)) * sizeof(float);

  // Allocate host arrays
  float* h_in = allocateHostArray(N * sizeof(float));
  float* h_out = allocateHostArray(N * sizeof(float));
  float* h_ref = allocateHostArray(N * sizeof(float));

  // Fill host array with random numbers
  srand(73);
  for (unsigned int i = 0; i < N; i++)
    // h_in[i] = ((double)rand() / RAND_MAX - 0.5f) * 2 * M;
    h_in[i] = (int)(rand() % MAX_RAND);
  // h_in[i] = 1.f;

  // Allocate device arrays
  float* d_arr[tree_depth + 1];
  for (int i = 0; i < tree_depth + 1; i++)
    d_arr[i] = allocateDeviceArray(sizeof(float) * lengths[i]);

  // Perform prescan on the device a number of times
  while (dur_in_total < dur_max) {
    nruns_gpu++;
    prescanOnDevice(h_in, h_out, d_arr, N, tree_depth, lengths, dimBlock,
                    dimGrid, shared_size, dur_ex, dur_in);
    dur_ex_total += dur_ex;
    dur_in_total += dur_in;
    if (dur_ex < dur_ex_min) dur_ex_min = dur_ex;
    if (dur_in < dur_in_min) dur_in_min = dur_in;
    if (dur_in_total == 0.f) break;
  }

  // Perform prescan on the host a number of times
  while (dur_cpu_total < dur_max) {
    nruns_cpu++;
    prescanOnHost(h_in, h_ref, N, dur_cpu);
    dur_cpu_total += dur_cpu;
    if (dur_cpu < dur_cpu_min) dur_cpu_min = dur_cpu;
    if (dur_cpu_total == 0.f) break;
  }

  dur_ex = dur_ex_total / nruns_gpu;
  dur_in = dur_in_total / nruns_gpu;
  dur_cpu = dur_cpu_total / nruns_cpu;

  // Compare device and host results
  float eps = (float)MAX_RAND * 0.001f;
  unsigned int nDiffs = checkResults(h_out, h_ref, N, eps);
  if (nDiffs == 0)
    printf("Test PASSED\n");
  else
    printf("Test FAILED; %u differences\n", nDiffs);

  // Print stuff
  printf("N: %u\n", N);
  printf("Tree depth: %d\n", tree_depth);
  printf("Block sizes: %d", dimBlock[0].x);
  for (int i = 1; i < tree_depth; i++) printf(", %d", dimBlock[i].x);
  printf("\n");
  printf("Grid sizes: %d", dimGrid[0].x);
  for (int i = 1; i < tree_depth; i++) printf(", %d", dimGrid[i].x);
  printf("\n");
  printf("GPU array lengths: %d", lengths[0]);
  for (int i = 1; i < tree_depth + 1; i++) printf(", %d", lengths[i]);
  printf("\n");
  printf("GPU last element: %24.14f\n", h_out[N - 1]);
  printf("CPU last element: %24.14f\n", h_ref[N - 1]);
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
