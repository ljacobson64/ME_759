#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16

// Kernel
__global__ void cudaMultiplyArrays(int* dA, int* dB, int* dC,
    int hA, int wA, int hB, int wB, int hC, int wC) {
  int y = blockIdx.y*BLOCK_SIZE + threadIdx.y;  // row
  int x = blockIdx.x*BLOCK_SIZE + threadIdx.x;  // column

  if (y >= hA || x >= wB) return;

  int result = 0;
  for (unsigned int i = 0; i < wA; i++) {
    result += dA[y*wA + i]*dB[i*wB + x];
  }

  dC[y*wC + x] = result;
}

// Kernel using shared memory
__global__ void cudaMultiplyArraysShared(int* dA, int* dB, int* dC,
    int hA, int wA, int hB, int wB, int hC, int wC) {
  // Thread and block indices
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Number of subarrays for each block
  int nsubs = (wA + BLOCK_SIZE - 1)/BLOCK_SIZE;

  // Initialize subarrays in shared memory
  __shared__ int sdA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int sdB[BLOCK_SIZE][BLOCK_SIZE];

  // Loop over each subarray
  int result = 0;
  for (unsigned int r = 0; r < nsubs; r++) {
    // Fill the subarrays in shared memory
    sdA[ty][tx] = dA[(by*BLOCK_SIZE + ty)*wA + ( r*BLOCK_SIZE + tx)];
    sdB[ty][tx] = dB[( r*BLOCK_SIZE + ty)*wB + (bx*BLOCK_SIZE + tx)];

    __syncthreads();

    // Don't add out of bounds elements
    int s_max;
    if ((r+1)*BLOCK_SIZE > wA) {
      s_max = wA - r*BLOCK_SIZE;
    } else {
      s_max = BLOCK_SIZE;
    }

    for (unsigned int s = 0; s < s_max; s++) {
      result += sdA[ty][s]*sdB[s][tx];
    }

    __syncthreads();
  }

  // Don't fill out of bounds elements
  if (bx*BLOCK_SIZE + tx >= wC) return;
  if (by*BLOCK_SIZE + ty >= hC) return;

  // Fill result array
  dC[(by*BLOCK_SIZE + ty)*wB + (bx*BLOCK_SIZE + tx)] = result;
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

void fill_array(int* A, int hA, int wA) {
  for (unsigned int i = 0; i < hA; i++) {
    for (unsigned int j = 0; j < wA; j++) {
      A[i*wA + j] = i + j;
    }
  }
}

int main(int argc, char *argv[]) {
  int m, n, p, nruns;
  bool shared, prt;
  if (argc == 1) {
    m = 16;
    n = 32;
    p = 1;
    nruns = 65536;
    shared = false;
    prt = true;
  } else if (argc == 5) {
    int siz = int_power(2, atoi(argv[1]));
    m = siz;
    n = siz;
    p = siz;
    nruns = int_power(2, atoi(argv[2]));
    if (atoi(argv[3]) > 0) shared = true;
    else shared = false;
    if (atoi(argv[4]) > 0) prt = true;
    else prt = false;
  }

  // Array sizes
  int hA = m, wA = n;
  int hB = n, wB = p;
  int hC = m, wC = p;
  int sA = hA*wA;
  int sB = hB*wB;
  int sC = hC*wC;

  // Allocate host arrays
  int *A, *B, *C;
  A = (int*)malloc(sizeof(int)*sA);
  B = (int*)malloc(sizeof(int)*sB);
  C = (int*)malloc(sizeof(int)*sC);

  // Allocate device arrays
  int *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(int)*sA);
  cudaMalloc(&dB, sizeof(int)*sB);
  cudaMalloc(&dC, sizeof(int)*sC);

  // Fill A and B with some integers
  fill_array(A, hA, wA);
  fill_array(B, hB, wB);

  // Set up block grid
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((wC + BLOCK_SIZE - 1)/BLOCK_SIZE,
               (hC + BLOCK_SIZE - 1)/BLOCK_SIZE);

  // Set up timing
  struct timespec start_in, end_in;
  long dur_in_ns;
  double dur_in = 0.0, dur_in_total = 0.0;
  double dur_in_min = 1e99, dur_in_max = 0.0;

  for (int i = 0; i < nruns; i++) {
    // Start inclusive timing
    clock_gettime(CLOCK_MONOTONIC, &start_in);

    // Copy host arrays to the device
    cudaMemcpy(dA, A, sizeof(int)*sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(int)*sB, cudaMemcpyHostToDevice);

    if (shared) {
      // Invoke the device kernel which multiplies the arrays with shared memory
      cudaMultiplyArraysShared<<<dimGrid, dimBlock>>>
          (dA, dB, dC, hA, wA, hB, wB, hC, wC);
    } else {
      // Invoke the device kernel which multiplies the arrays
      cudaMultiplyArrays<<<dimGrid, dimBlock>>>
          (dA, dB, dC, hA, wA, hB, wB, hC, wC);
    }

    // Copy the result array back to the host
    cudaMemcpy(C, dC, sizeof(int)*sC, cudaMemcpyDeviceToHost);

    // End inclusive timing
    clock_gettime(CLOCK_MONOTONIC, &end_in);

    // Calculate duration
    dur_in_ns = (end_in.tv_sec - start_in.tv_sec)*1000000000l +
                 end_in.tv_nsec - start_in.tv_nsec;
    dur_in = (double)(dur_in_ns/1000000.0);
    dur_in_total += dur_in;
    if (dur_in < dur_in_min) dur_in_min = dur_in;
    if (dur_in > dur_in_max) dur_in_max = dur_in;
  }

  // Write result to file
  if (prt) {
    FILE *fp;
    fp = fopen("problem2.out", "w");
    for (int i = 0; i < hC; i++) {
      for (int j = 0; j < wC; j++) {
        fprintf(fp, "%12d ", C[i*wC + j]);
      }
      fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    fclose(fp);
  }

  // Free memory
  free(A);
  free(B);
  free(C);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  // Get device properties
  cudaDeviceProp gpu_props;
  cudaGetDeviceProperties(&gpu_props, 0);

  // Print some information
  printf("Device name: %s\n", gpu_props.name);
  printf("Dimension 1 (m):      %12d\n", m);
  printf("Dimension 2 (n):      %12d\n", n);
  printf("Dimension 3 (p):      %12d\n", p);
  printf("Block size:           %12d\n", BLOCK_SIZE);
  printf("Number of runs:       %12d\n", nruns);
  printf("Using shared memory?: %12s\n", shared ? "True" : "False");
  printf("Inclusive time (min): %12.6f ms\n", dur_in_min);
  printf("Inclusive time (avg): %12.6f ms\n", dur_in_total/nruns);
  printf("Inclusive time (max): %12.6f ms\n", dur_in_max);
  printf("\n");

  return 0;
}
