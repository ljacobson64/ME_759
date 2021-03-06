#include "cuda.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#include "thrust/scan.h"

#define BLOCK_SIZE 512
#define DOUBLE_BLOCK 1024
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(x) ((x) >> LOG_NUM_BANKS)

__global__ void prescanKernel(double* d_in, double* d_out, double* d_sums,
                              unsigned int N) {
  extern __shared__ double s_data[];
  unsigned int shared_end =
      DOUBLE_BLOCK + CONFLICT_FREE_OFFSET(DOUBLE_BLOCK) - 2;

  // Indexing
  unsigned int offset = 1;
  unsigned int iblock = gridDim.x * blockIdx.y + blockIdx.x;
  unsigned int ai = threadIdx.x;
  unsigned int bi = threadIdx.x + BLOCK_SIZE;
  unsigned int ag = ai + iblock * DOUBLE_BLOCK;
  unsigned int bg = bi + iblock * DOUBLE_BLOCK;
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
    d_sums[iblock] = s_data[shared_end];
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

      double temp = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += temp;
    }
  }

  // Write results to global memory
  __syncthreads();
  if (ag < N) d_out[ag] = s_data[ai + bankOffsetA];
  if (bg < N) d_out[bg] = s_data[bi + bankOffsetB];
}

__global__ void additionKernel(double* d_in, double* d_out, double* d_sums,
                               unsigned int N) {
  unsigned int iblock = gridDim.x * blockIdx.y + blockIdx.x;
  unsigned int i = iblock * DOUBLE_BLOCK + threadIdx.x;
  if (i < N) d_out[i] = d_in[i] + d_sums[iblock];
  if (i + BLOCK_SIZE < N)
    d_out[i + BLOCK_SIZE] = d_in[i + BLOCK_SIZE] + d_sums[iblock];
}

void prescanDevice(double* h_in, double* h_out, double** d_arr, unsigned int N,
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
  cudaMemcpy(d_arr[0], h_in, lengths[0] * sizeof(double),
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
  cudaMemcpy(h_out, d_arr[0], lengths[0] * sizeof(double),
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

void prescanThrust(double* h_in, double* h_out, unsigned int N, float& dur) {
  // Setup timing
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // Perform reduction on the device using thrust
  cudaEventRecord(start, 0);
  thrust::exclusive_scan(h_in, h_in + N, h_out);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  // Calculate duration
  cudaEventElapsedTime(&dur, start, end);

  // Cleanup timing
  cudaEventDestroy(start);
  cudaEventDestroy(end);
}

void prescanHost(double* h_in, double* h_out, unsigned int N, float& dur) {
  // Setup timing
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // Perform prescan on host
  cudaEventRecord(start, 0);
  h_out[0] = 0.f;
  for (unsigned int i = 1; i < N; i++) h_out[i] = h_in[i - 1] + h_out[i - 1];
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  // Calculate duration
  cudaEventElapsedTime(&dur, start, end);

  // Cleanup timing
  cudaEventDestroy(start);
  cudaEventDestroy(end);
}

unsigned int checkResults(double* h_out, double* h_ref, unsigned int N,
                          double eps) {
  unsigned int nDiffs = 0;
  for (unsigned int i = 0; i < N; i++) {
    double delta = abs(h_out[i] - h_ref[i]);
    if (delta > eps) nDiffs++;
  }
  return nDiffs;
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
  printf("Usage: ./p2 [[M N] [dur_max]]\n");
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
  double* h_in = allocateHostArray(N * sizeof(double));
  double* h_device = allocateHostArray(N * sizeof(double));
  double* h_thrust = allocateHostArray(N * sizeof(double));
  double* h_cpu = allocateHostArray(N * sizeof(double));

  // Setup host array and fill with random numbers
  srand(73);
  for (unsigned int i = 0; i < N; i++)
    h_in[i] = ((double)rand() / RAND_MAX - 0.5f) * 2 * M;

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
    if (lengths[i + 1] < 32768)
      dimGrid[i].x = lengths[i + 1];
    else {
      dimGrid[i].x = 32768;
      dimGrid[i].y = (lengths[i + 1] + 32768 - 1) / 32768;
    }
  }

  // Shared memory size
  unsigned int shared_size =
      (DOUBLE_BLOCK + CONFLICT_FREE_OFFSET(DOUBLE_BLOCK)) * sizeof(double);

  // Allocate device arrays
  double* d_arr[tree_depth + 1];
  for (int i = 0; i < tree_depth + 1; i++)
    d_arr[i] = allocateDeviceArray(lengths[i] * sizeof(double));

  // Setup timing
  unsigned int nruns_device = 0;
  unsigned int nruns_thrust = 0;
  unsigned int nruns_cpu = 0;
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
    prescanDevice(h_in, h_device, d_arr, N, tree_depth, lengths, dimBlock,
                  dimGrid, shared_size, dur_ex, dur_in);
    dur_ex_total += dur_ex;
    dur_in_total += dur_in;
    if (dur_ex < dur_ex_min) dur_ex_min = dur_ex;
    if (dur_in < dur_in_min) dur_in_min = dur_in;
    if (dur_in_total <= 0.f) break;
  }

  // Vector reduction on the device with thrust
  while (dur_thrust_total < dur_max) {
    nruns_thrust++;
    prescanThrust(h_in, h_thrust, N, dur_thrust);
    dur_thrust_total += dur_thrust;
    if (dur_thrust < dur_thrust_min) dur_thrust_min = dur_thrust;
    if (dur_thrust_total <= 0.f) break;
  }

  // Vector reduction on CPU
  while (dur_cpu_total < dur_max) {
    nruns_cpu++;
    prescanHost(h_in, h_cpu, N, dur_cpu);
    dur_cpu_total += dur_cpu;
    if (dur_cpu < dur_cpu_min) dur_cpu_min = dur_cpu;
    if (dur_cpu_total <= 0.f) break;
  }

  dur_ex = dur_ex_total / nruns_device;
  dur_in = dur_in_total / nruns_device;
  dur_thrust = dur_thrust_total / nruns_thrust;
  dur_cpu = dur_cpu_total / nruns_cpu;

  // Compare device and host results
  double eps = (double)M * 0.001f;
  unsigned int nDiffs_device = checkResults(h_device, h_cpu, N, eps);
  unsigned int nDiffs_thrust = checkResults(h_thrust, h_cpu, N, eps);
  if (nDiffs_device == 0)
    printf("Test PASSED (device)\n");
  else
    printf("Test FAILED (device); %u differences\n", nDiffs_device);
  if (nDiffs_thrust == 0)
    printf("Test PASSED (thrust)\n");
  else
    printf("Test FAILED (thrust); %u differences\n", nDiffs_thrust);

  // Print stuff
  printf("N: %u\n", N);
  printf("M: %u\n", M);
  printf("Tree depth: %d\n", tree_depth);
  printf("Block sizes: %dx%d", dimBlock[0].y, dimBlock[0].x);
  for (int i = 1; i < tree_depth; i++)
    printf(", %dx%d", dimBlock[i].y, dimBlock[i].x);
  printf("\n");
  printf("Grid sizes: %dx%d", dimGrid[0].y, dimGrid[0].x);
  for (int i = 1; i < tree_depth; i++)
    printf(", %dx%d", dimGrid[i].y, dimGrid[i].x);
  printf("\n");
  printf("GPU array lengths: %d", lengths[0]);
  for (int i = 1; i < tree_depth + 1; i++) printf(", %d", lengths[i]);
  printf("\n");
  printf("Last element (device): %24.14f\n", h_device[N - 1]);
  printf("Last element (thrust): %24.14f\n", h_thrust[N - 1]);
  printf("Last element (CPU):    %24.14f\n", h_cpu[N - 1]);
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
  //for (int i = 0; i < tree_depth + 1; i++) cudaFree(d_arr[i]);

  return 0;
}
