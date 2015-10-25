#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 512
#define RADIUS 3

int checkResults(int startElem, int endElem, float *cudaRes, float *res) {
  int nDiffs = 0;
  const float smallVal = 0.000001f;
  for (int i = startElem; i < endElem; i++)
    if (fabs(cudaRes[i] - res[i]) > smallVal) nDiffs++;
  return nDiffs;
}

void initializeWeights(float *weights, int rad) {
  // Hardcoded for RADIUS = 3
  weights[0] = 0.50f;
  weights[1] = 0.75f;
  weights[2] = 1.25f;
  weights[3] = 2.00f;
  weights[4] = 1.25f;
  weights[5] = 0.75f;
  weights[6] = 0.50f;
}

void initializeArray(float *arr, int nElements) {
  const int myMinNumber = -5;
  const int myMaxNumber = 5;
  srand(time(NULL));
  for (int i = 0; i < nElements; i++)
    arr[i] = (float)(rand() % (myMaxNumber - myMinNumber + 1) + myMinNumber);
}

void applyStencil1D_SEQ(int sIdx, int eIdx, const float *weights, float *in,
                        float *out) {
  for (int i = sIdx; i < eIdx; i++) {
    out[i] = 0;
    // Loop over all elements in the stencil
    for (int j = -RADIUS; j <= RADIUS; j++)
      out[i] += weights[j + RADIUS] * in[i + j];
    out[i] = out[i] / (2 * RADIUS + 1);
  }
}

__global__ void applyStencil1D_V4(int sIdx, int eIdx, const float *weights,
                                  float *in, float *out) {
  int i = sIdx + blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= eIdx) return;

  float result = 0.f;
  result += weights[0] * in[i - 3];
  result += weights[1] * in[i - 2];
  result += weights[2] * in[i - 1];
  result += weights[3] * in[i];
  result += weights[4] * in[i + 1];
  result += weights[5] * in[i + 2];
  result += weights[6] * in[i + 3];
  result /= 7.f;
  out[i] = result;
}

__global__ void applyStencil1D_V5(int sIdx, int eIdx, const float *weights,
                                  float *in, float *out) {
  extern __shared__ float sdata[];
  int i = sIdx + blockIdx.x * blockDim.x + threadIdx.x;

  // Read into shared memory
  sdata[threadIdx.x + RADIUS] = in[i];
  if (threadIdx.x < RADIUS) {
    sdata[threadIdx.x] = in[i - RADIUS];
    sdata[threadIdx.x + RADIUS + BLOCK_SIZE] = in[i + BLOCK_SIZE];
  }

  __syncthreads();

  if (i >= eIdx) return;

  // Calculate result
  float result = 0.f;
  result += weights[0] * sdata[threadIdx.x];
  result += weights[1] * sdata[threadIdx.x + 1];
  result += weights[2] * sdata[threadIdx.x + 2];
  result += weights[3] * sdata[threadIdx.x + 3];
  result += weights[4] * sdata[threadIdx.x + 4];
  result += weights[5] * sdata[threadIdx.x + 5];
  result += weights[6] * sdata[threadIdx.x + 6];
  result /= 7.f;
  out[i] = result;
}

int int_power(int x, int n) {
  if (n <= 0) return 1;
  int y = 1;
  while (n > 1) {
    if (n % 2 == 0) {
      x *= x;
      n /= 2;
    } else {
      y *= x;
      x *= x;
      n = (n - 1) / 2;
    }
  }
  return x * y;
}

int main(int argc, char *argv[]) {
  int version, N;
  if (argc == 3) {
    version = atoi(argv[1]);
    N = int_power(2, atoi(argv[2]));
  } else {
    printf("Usage: ./p1 <kernel_version> <log2(N)>\n");
    printf("Allowed versions: 4, 5, 6\n");
    return 0;
  }

  printf("Version: %d\n", version);
  printf("N: 2^%d = %d\n", atoi(argv[2]), N);

  int size = N * sizeof(float);
  int wsize = (2 * RADIUS + 1) * sizeof(float);

  // Setup timing
  float dur_ex, dur_in, dur_cpu;
  cudaEvent_t start_ex, end_ex, start_in, end_in, start_cpu, end_cpu;
  cudaEventCreate(&start_ex);
  cudaEventCreate(&end_ex);
  cudaEventCreate(&start_in);
  cudaEventCreate(&end_in);
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&end_cpu);

  // Allocate host resources
  float *weights, *in, *out, *cuda_out;
  if (version == 4 || version == 5) {
    weights = (float *)malloc(wsize);
    in = (float *)malloc(size);
    out = (float *)malloc(size);
    cuda_out = (float *)malloc(size);
  } else if (version == 6) {
    cudaMallocHost(&weights, wsize);
    cudaMallocHost(&in, size);
    cudaMallocHost(&out, size);
    cudaMallocHost(&cuda_out, size);
  }

  // Allocate device resources
  float *d_weights, *d_in, *d_out;
  cudaMalloc(&d_weights, wsize);
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // Fill weights and array
  initializeWeights(weights, RADIUS);
  initializeArray(in, N);

  // Setup grid
  dim3 dimBlock = BLOCK_SIZE;
  dim3 dimGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int shared_size = (BLOCK_SIZE + 2 * RADIUS) * sizeof(float);

  // Start inclusive timing
  cudaEventRecord(start_in, 0);

  // Copy to device
  cudaMemcpy(d_weights, weights, wsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

  // Start exclusive timing
  cudaEventRecord(start_ex, 0);

  // Execute kernel
  if (version == 4)
    applyStencil1D_V4 <<<dimGrid, dimBlock>>>
        (RADIUS, N - RADIUS, d_weights, d_in, d_out);
  else if (version == 5 || version == 6)
    applyStencil1D_V5 <<<dimGrid, dimBlock, shared_size>>>
        (RADIUS, N - RADIUS, d_weights, d_in, d_out);

  // End exclusive timing
  cudaEventRecord(end_ex, 0);
  cudaEventSynchronize(end_ex);

  // Copy from device
  cudaMemcpy(cuda_out, d_out, size, cudaMemcpyDeviceToHost);

  // End inclusive timing
  cudaEventRecord(end_in, 0);
  cudaEventSynchronize(end_in);

  // Check against result on CPU
  cudaEventRecord(start_cpu, 0);
  applyStencil1D_SEQ(RADIUS, N - RADIUS, weights, in, out);
  cudaEventRecord(end_cpu, 0);
  cudaEventSynchronize(end_cpu);
  int nDiffs = checkResults(RADIUS, N - RADIUS, cuda_out, out);
  if (nDiffs == 0) printf("Looks good.\n");
  else printf("Doesn't look good: %d differences\n", nDiffs);

  // Calculate durations
  cudaEventElapsedTime(&dur_ex, start_ex, end_ex);
  cudaEventElapsedTime(&dur_in, start_in, end_in);
  cudaEventElapsedTime(&dur_cpu, start_cpu, end_cpu);
  printf("GPU execution time (exclusive): %15.6f ms\n", dur_ex);
  printf("GPU execution time (inclusive): %15.6f ms\n", dur_in);
  printf("CPU execution time:             %15.6f ms\n", dur_cpu);
  printf("\n");

  // Free resources
  free(weights);
  free(in);
  free(out);
  free(cuda_out);
  cudaFree(d_weights);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaEventDestroy(start_ex);
  cudaEventDestroy(end_ex);
  cudaEventDestroy(start_in);
  cudaEventDestroy(end_in);
  cudaEventDestroy(start_cpu);
  cudaEventDestroy(end_cpu);

  return 0;
}
