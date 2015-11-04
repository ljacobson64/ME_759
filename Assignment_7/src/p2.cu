#include "cuda.h"
#include "limits.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#define BLOCK_SIZE 512
#define ELEMS_PER_THREAD 192

template <unsigned int blockSize>
__device__ void warpReduce(volatile double* s_data, unsigned int t) {
  if (blockSize >= 64) s_data[t] += s_data[t + 32];
  if (blockSize >= 32) s_data[t] += s_data[t + 16];
  if (blockSize >= 16) s_data[t] += s_data[t + 8];
  if (blockSize >= 8) s_data[t] += s_data[t + 4];
  if (blockSize >= 4) s_data[t] += s_data[t + 2];
  if (blockSize >= 2) s_data[t] += s_data[t + 1];
}

template <unsigned int blockSize>
__global__ void reductionKernel(double* d_in, double* d_out, unsigned int N) {
  extern __shared__ double s_data[];

  // Indexing
  unsigned int t = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + t;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  // Load some elements into shared memory
  s_data[t] = 0.f;
  while (i + blockSize < N) {
    s_data[t] += d_in[i] + d_in[i + blockDim.x];
    i += gridSize;
  }
  if (i < N) s_data[t] += d_in[i];
  __syncthreads();

  // Unroll the loop
  if (blockSize >= 512) {
    if (t < 256) s_data[t] += s_data[t + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (t < 128) s_data[t] += s_data[t + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (t < 64) s_data[t] += s_data[t + 64];
    __syncthreads();
  }

  if (t < 32) warpReduce<blockSize>(s_data, t);

  // Write the result for each block into d_out
  if (t == 0) d_out[blockIdx.x] = s_data[0];
}

void reductionOnDevice(double* h_in, double* h_out, double** d_arr,
                       unsigned int N, int tree_depth, unsigned int* lengths,
                       dim3* dimBlock, dim3* dimGrid, unsigned int shared_size,
                       float& dur_gpu) {
  // Setup CUDA streams
  unsigned int stream_lengths[2];
  unsigned int stream_sizes[2];
  stream_lengths[0] = ((N / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  stream_lengths[1] = N - stream_lengths[0];
  for (int i = 0; i < 2; i++)
    stream_sizes[i] = sizeof(double) * stream_lengths[i];
  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);

  // Setup timing
  cudaEvent_t start_gpu, end_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&end_gpu);

  // Copy host array to device (stream0)
  cudaEventRecord(start_gpu, 0);
  cudaMemcpyAsync(d_arr[0], h_in, stream_sizes[0], cudaMemcpyHostToDevice, stream0);

  // Perform reduction on device (stream0)
  reductionKernel<BLOCK_SIZE> <<<(stream_lengths[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_size, stream0>>>
      (d_arr[0], d_arr[1], stream_lengths[0]);

  // Copy host array to device (stream1)
  cudaMemcpyAsync(d_arr[0] + stream_sizes[0], h_in + stream_sizes[0], stream_lengths[1], cudaMemcpyHostToDevice, stream1);

  // Perform reduction on device (stream1)
  reductionKernel<BLOCK_SIZE> <<<(stream_lengths[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_size, stream1>>>
      (d_arr[0] + stream_sizes[0], d_arr[1], stream_lengths[1]);

  // Perform second reduction on device (stream1)
  reductionKernel<BLOCK_SIZE> <<<dimGrid[1], dimBlock[1], shared_size, stream1>>>
      (d_arr[1], d_arr[2], lengths[1]);

  // Copy host array to device (stream1)
  cudaMemcpyAsync(h_out, d_arr[tree_depth], sizeof(double), cudaMemcpyDeviceToHost, stream1);
  cudaEventRecord(end_gpu, 0);
  cudaEventSynchronize(end_gpu);

  // Calculate durations
  cudaEventElapsedTime(&dur_gpu, start_gpu, end_gpu);
}

void reductionOnHost(double* h_in, double* h_ref, unsigned int N,
                     float& dur_cpu) {
  // Setup timing
  cudaEvent_t start_cpu, end_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&end_cpu);

  // Perform reduction on host
  cudaEventRecord(start_cpu, 0);
  double result = 0.f;
  for (unsigned int i = 0; i < N; i++) result += h_in[i];
  *h_ref = result;
  cudaEventRecord(end_cpu, 0);
  cudaEventSynchronize(end_cpu);

  // Calculate duration
  cudaEventElapsedTime(&dur_cpu, start_cpu, end_cpu);
}

bool checkResults(double* h_out, double* h_ref, double eps) {
  double delta = abs(*h_out - *h_ref);
  if (delta > eps) return false;
  return true;
}

void exitUsage() {
  printf("Usage: ./p2 [<M> <N> [<dur_max>]]\n");
  exit(EXIT_SUCCESS);
}

void parseInput(int argc, char* argv[], unsigned int& N, unsigned int& M,
                float& dur_max) {
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
  dur_max *= 1000;
}

double* allocateHostArray(unsigned int size) {
  double* h_array;
  cudaError_t code = cudaHostAlloc(&h_array, size, cudaHostAllocDefault);
  if (code != cudaSuccess) {
    printf("Memory allocation on the host was unsuccessful.\n");
    exit(EXIT_FAILURE);
  }
  return h_array;
}

double* allocateDeviceArray(unsigned int size) {
  double* d_arr;
  cudaError_t code = cudaMalloc(&d_arr, size);
  if (code != cudaSuccess) {
    printf("Memory allocation on the device was unsuccessful.\n");
    exit(EXIT_FAILURE);
  }
  return d_arr;
}

int main(int argc, char* argv[]) {
  unsigned int N, M;
  float dur_max;
  parseInput(argc, argv, N, M, dur_max);

  // Calculate the tree depth
  int tree_depth = 0;
  {
    unsigned int length = N;
    while (length > 1) {
      length = (length + (BLOCK_SIZE * ELEMS_PER_THREAD) - 1) /
               (BLOCK_SIZE * ELEMS_PER_THREAD);
      tree_depth++;
    }
  }

  // Calculate the lengths of the device arrays
  unsigned int lengths[tree_depth + 1];
  lengths[0] = N;
  for (int i = 1; i < tree_depth + 1; i++)
    lengths[i] = (lengths[i - 1] + (BLOCK_SIZE * ELEMS_PER_THREAD) - 1) /
                 (BLOCK_SIZE * ELEMS_PER_THREAD);

  // Setup grid
  dim3 dimBlock[tree_depth];
  dim3 dimGrid[tree_depth];
  for (int i = 0; i < tree_depth; i++) {
    dimBlock[i].x = BLOCK_SIZE;
    dimGrid[i].x = lengths[i + 1];
  }

  // Shared memory size
  unsigned int shared_size = sizeof(double) * BLOCK_SIZE;

  // Allocate host arrays
  double* h_in = allocateHostArray(sizeof(double) * N);
  double* h_out = allocateHostArray(sizeof(double));
  double* h_ref = allocateHostArray(sizeof(double));

  // Fill host array with random numbers
  srand(73);
  for (unsigned int i = 0; i < N; i++)
    h_in[i] = ((double)rand() / RAND_MAX - 0.5f) * 2 * M;

  // Allocate device arrays
  double* d_arr[tree_depth + 1];
  for (int i = 0; i < tree_depth + 1; i++)
    d_arr[i] = allocateDeviceArray(sizeof(double) * lengths[i]);

  // Setup timing
  int nruns_gpu = 0;
  int nruns_cpu = 0;
  float dur_gpu, dur_cpu;
  float dur_gpu_total = 0.f;
  float dur_cpu_total = 0.f;
  float dur_gpu_min = 1e99;
  float dur_cpu_min = 1e99;

  // Perform reduction on the device a number of times
  while (dur_gpu_total < dur_max) {
    nruns_gpu++;
    reductionOnDevice(h_in, h_out, d_arr, N, tree_depth, lengths, dimBlock,
                      dimGrid, shared_size, dur_gpu);
    printf("%12.6f\n", dur_gpu);
    dur_gpu_total += dur_gpu;
    if (dur_gpu < dur_gpu_min) dur_gpu_min = dur_gpu;
    if (dur_gpu_total == 0.f) break;
  }

  // Perform reduction on the host a number of times
  while (dur_cpu_total < dur_max) {
    nruns_cpu++;
    reductionOnHost(h_in, h_ref, N, dur_cpu);
    dur_cpu_total += dur_cpu;
    if (dur_cpu < dur_cpu_min) dur_cpu_min = dur_cpu;
    if (dur_cpu_total == 0.f) break;
  }

  dur_gpu = dur_gpu_total / nruns_gpu;
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
  printf("Elements per thread: %d\n", ELEMS_PER_THREAD);
  printf("Tree depth: %d\n", tree_depth);
  printf("Block sizes: ");
  for (int i = 0; i < tree_depth - 1; i++) printf("%d, ", dimBlock[i].x);
  printf("%d\n", dimBlock[tree_depth - 1].x);
  printf("Grid sizes: ");
  for (int i = 0; i < tree_depth - 1; i++) printf("%d, ", dimGrid[i].x);
  printf("%d\n", dimGrid[tree_depth - 1].x);
  printf("GPU array lengths: ");
  for (int i = 0; i < tree_depth; i++) printf("%d, ", lengths[i]);
  printf("%d\n", lengths[tree_depth]);
  printf("GPU result: %24.14f\n", *h_out);
  printf("CPU result: %24.14f\n", *h_ref);
  printf("Timing results  %12s %12s %8s\n", "Average", "Minimum", "Num_runs");
  printf("GPU:            %12.6f %12.6f %8d\n", dur_gpu, dur_gpu_min, nruns_gpu);
  printf("CPU:            %12.6f %12.6f %8d\n", dur_cpu, dur_cpu_min, nruns_cpu);
  printf("\n");

  // Free memory
  cudaFreeHost(h_in);
  cudaFreeHost(h_out);
  cudaFreeHost(h_ref);
  for (int i = 0; i < tree_depth + 1; i++)
    cudaFree(d_arr[i]);

  return 0;
}
