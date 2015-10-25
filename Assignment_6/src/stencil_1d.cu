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
      n = (n-1)/2;
    }
  }
  return x*y;
}

int main(int argc, char *argv[]) {
  int method, N;
  if (argc == 3) {
    method = atoi(argv[1]);
    N = int_power(10, atoi(argv[2]));
  }
  else {
    printf("Usage: ./p1 <kernel_version> <log10(N)>\n");
    printf("Allowed versions: 4, 5\n");
    return 0;
  }

  int size = N * sizeof(float);
  int wsize = (2 * RADIUS + 1) * sizeof(float);
  
  // Allocate host resources
  float *weights, *in, *out, *cuda_out;
  weights  = (float*)malloc(wsize);
  in       = (float*)malloc(size);
  out      = (float*)malloc(size);
  cuda_out = (float*)malloc(size);

  // Allocate device resources
  float *d_weights, *d_in, *d_out;
  cudaMalloc(&d_weights, wsize);
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // Fill weights and array
  initializeWeights(weights, RADIUS);
  initializeArray(in, N);

  // Copy to device
  cudaMemcpy(d_weights, weights, wsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

  // Execute kernel
  if (method == 4)
    applyStencil1D_V4 <<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
        (RADIUS, N - RADIUS, d_weights, d_in, d_out);
  else if (method == 5)
    applyStencil1D_V5 <<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE,
        (BLOCK_SIZE + 2 * RADIUS) * sizeof(float)>>>
        (RADIUS, N - RADIUS, d_weights, d_in, d_out);


  // Copy from device
  cudaMemcpy(cuda_out, d_out, size, cudaMemcpyDeviceToHost);

  // Check against result on CPU
  applyStencil1D_SEQ(RADIUS, N - RADIUS, weights, in, out);
  int nDiffs = checkResults(RADIUS, N - RADIUS, cuda_out, out);
  if (nDiffs == 0) printf("Looks good.\n");
  else printf("Doesn't look good: %d differences\n", nDiffs);

  // Free resources
  free(weights);
  free(in);
  free(out);
  free(cuda_out);
  cudaFree(d_weights);
  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
