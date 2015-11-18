#include "cuda.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#include "thrust/device_vector.h"

#define BLOCK_SIZE 512
#define ELEMS_PER_THREAD 32

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
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  // Load some elements into shared memory
  s_data[t] = 0.f;
  while (i + blockSize < N) {
    s_data[t] += d_in[i] + d_in[i + blockSize];
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

void reductionDevice(double* h_in, double* h_out, double** d_arr,
                     unsigned int N, int tree_depth, unsigned int* lengths,
                     dim3* dimBlock, dim3* dimGrid, unsigned int& shared_size,
                     float& dur_ex, float& dur_in) {
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
    reductionKernel<BLOCK_SIZE> <<<dimGrid[i], dimBlock[i], shared_size>>>
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
}

void reductionThrust(double* h_in, double* h_out, unsigned int N,
                     float& dur_thrust) {
  // Setup timing
  cudaEvent_t start_thrust, end_thrust;
  cudaEventCreate(&start_thrust);
  cudaEventCreate(&end_thrust);

  // Perform reduction on the device using thrust
  cudaEventRecord(start_thrust, 0);
  *h_out = thrust::reduce(h_in, h_in + N);
  cudaEventRecord(end_thrust, 0);
  cudaEventSynchronize(end_thrust);

  // Calculate duration
  cudaEventElapsedTime(&dur_thrust, start_thrust, end_thrust);
}

void reductionHost(double* h_in, double* h_out, unsigned int N,
                   float& dur_cpu) {
  // Setup timing
  cudaEvent_t start_cpu, end_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&end_cpu);

  // Perform reduction on host
  cudaEventRecord(start_cpu, 0);
  double result = 0.f;
  for (unsigned int i = 0; i < N; i++) result += h_in[i];
  *h_out = result;
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

double* allocateHostArray(unsigned int size) {
  double* h_array;
  cudaError_t code = cudaMallocHost(&h_array, size);
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

void exitUsage() {
  printf("Usage: ./p2 [<M> <N> [<dur_max>]]\n");
  exit(EXIT_SUCCESS);
}

void parseInput(int argc, char** argv, unsigned int& N, unsigned int& M,
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

int main(int argc, char** argv) {
  // Parse command line arguments
  unsigned int N, M;
  float dur_max;
  parseInput(argc, argv, N, M, dur_max);

  // Allocate host arrays
  double* h_in = allocateHostArray(sizeof(double) * N);
  double* h_device = allocateHostArray(sizeof(double));
  double* h_thrust = allocateHostArray(sizeof(double));
  double* h_cpu = allocateHostArray(sizeof(double));

  // Setup host array and fill with random numbers
  srand(73);
  for (unsigned int i = 0; i < N; i++)
    h_in[i] = ((double)rand() / RAND_MAX - 0.5f) * 2 * M;

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
  unsigned int shared_size = BLOCK_SIZE * sizeof(double);

  // Allocate device arrays
  double* d_arr[tree_depth + 1];
  for (int i = 0; i < tree_depth + 1; i++)
    d_arr[i] = allocateDeviceArray(sizeof(double) * lengths[i]);

  // Setup timing
  int nruns_device = 0;
  int nruns_thrust = 0;
  int nruns_cpu = 0;
  float dur_ex, dur_in, dur_thrust, dur_cpu;
  float dur_ex_total = 0.f;
  float dur_in_total = 0.f;
  float dur_thrust_total = 0.f;
  float dur_cpu_total = 0.f;
  float dur_ex_min = 1e99;
  float dur_in_min = 1e99;
  float dur_thrust_min = 1e99;
  float dur_cpu_min = 1e99;

  // Vector reduction on the device
  while (dur_in_total < dur_max) {
    nruns_device++;
    reductionDevice(h_in, h_device, d_arr, N, tree_depth, lengths, dimBlock,
                    dimGrid, shared_size, dur_ex, dur_in);
    dur_ex_total += dur_ex;
    dur_in_total += dur_in;
    if (dur_ex < dur_ex_min) dur_ex_min = dur_ex;
    if (dur_in < dur_in_min) dur_in_min = dur_in;
    if (dur_in_total == 0.f) break;
  }

  // Vector reduction on the device with thrust
  while (dur_thrust_total < dur_max) {
    nruns_thrust++;
    reductionThrust(h_in, h_thrust, N, dur_thrust);
    dur_thrust_total += dur_thrust;
    if (dur_thrust < dur_thrust_min) dur_thrust_min = dur_thrust;
    if (dur_thrust_total == 0.f) break;
  }

  // Vector reduction on CPU
  while (dur_cpu_total < dur_max) {
    nruns_cpu++;
    reductionHost(h_in, h_cpu, N, dur_cpu);
    dur_cpu_total += dur_cpu;
    if (dur_cpu < dur_cpu_min) dur_cpu_min = dur_cpu;
    if (dur_cpu_total == 0.f) break;
  }

  // Compare device and host results
  double eps = (double)M * 0.001f;
  bool passed_device = checkResults(h_device, h_cpu, eps);
  bool passed_thrust = checkResults(h_thrust, h_cpu, eps);
  if (passed_device)
    printf("Test PASSED (device)\n");
  else
    printf("Test FAILED (device)\n");
  if (passed_thrust)
    printf("Test PASSED (thrust)\n");
  else
    printf("Test FAILED (thrust)\n");

  // Print stuff
  printf("N: %u\n", N);
  printf("M: %u\n", M);
  printf("Elements per thread: %d\n", ELEMS_PER_THREAD);
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
  printf("Result (Device): %24.14f\n", *h_device);
  printf("Result (Thrust): %24.14f\n", *h_thrust);
  printf("Result (CPU):    %24.14f\n", *h_cpu);
  printf("Timing results  %12s  %12s  %8s\n", "Average", "Minimum", "Num_runs");
  printf("Device ex:      %12.6f  %12.6f  %8d\n", dur_ex, dur_ex_min,
         nruns_device);
  printf("Device in:      %12.6f  %12.6f  %8d\n", dur_in, dur_in_min,
         nruns_device);
  printf("Thrust:         %12.6f  %12.6f  %8d\n", dur_thrust, dur_thrust_min,
         nruns_thrust);
  printf("CPU:            %12.6f  %12.6f  %8d\n", dur_cpu, dur_cpu_min,
         nruns_cpu);
  printf("\n");

  // Free arrays
  cudaFree(h_in);
  cudaFree(h_device);
  cudaFree(h_thrust);
  cudaFree(h_cpu);
  for (int i = 0; i < tree_depth + 1; i++) cudaFree(d_arr[i]);

  return 0;
}
