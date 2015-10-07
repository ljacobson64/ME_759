#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16

// Kernel using global memory
__global__ void cudaMultiplyArraysGlobal(int* dA, int* dB, int* dC,
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

void fill_array(int* A, int hA, int wA) {
  for (unsigned int i = 0; i < hA; i++) {
    for (unsigned int j = 0; j < wA; j++) {
      A[i*wA + j] = i + j;
    }
  }
}

int main(int argc, char *argv[]) {
  int m, n, p, nruns, prt;
  if (argc == 1) {
    m = 16;
    n = 32;
    p = 1;
    nruns = 65536;
    prt = true;
  } else if (argc == 6) {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);
    nruns = atoi(argv[4]);
    if (atoi(argv[5]) > 0) prt = true;
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
  double dur_in_max = 0.0, dur_in_min = 1e99;

  for (int i = 0; i < nruns; i++) {
    // Start inclusive timing
    clock_gettime(CLOCK_MONOTONIC, &start_in);

    // Copy host arrays to the device
    cudaMemcpy(dA, A, sizeof(int)*sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(int)*sB, cudaMemcpyHostToDevice);

    // Invoke the device kernel which multiplies the arrays using global memory
    cudaMultiplyArraysGlobal<<<dimGrid, dimBlock>>>
        (dA, dB, dC, hA, wA, hB, wB, hC, wC);

    // Invoke the device kernel which multiplies the arrays using shared memory
    //cudaMultiplyArraysShared<<<dimGrid, dimBlock>>>
    //    (dA, dB, dC, hA, wA, hB, wB, hC, wC);

    // Copy the result array back to the host
    cudaMemcpy(C, dC, sizeof(int)*sC, cudaMemcpyDeviceToHost);

    // End inclusive timing
    clock_gettime(CLOCK_MONOTONIC, &end_in);

    // Calculate duration
    dur_in_ns = (end_in.tv_sec - start_in.tv_sec)*1000000000l +
                 end_in.tv_nsec - start_in.tv_nsec;
    dur_in = (double)(dur_in_ns/1000000.0);
    dur_in_total += dur_in;
    if (dur_in > dur_in_max) dur_in_max = dur_in;
    if (dur_in < dur_in_min) dur_in_min = dur_in;
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
  printf("Dimension 1 (m): %12d\n", m);
  printf("Dimension 2 (n): %12d\n", n);
  printf("Dimension 3 (p): %12d\n", p);
  printf("Block size:      %12d\n", BLOCK_SIZE);
  printf("Number of runs:  %12d\n", nruns);
  printf("Inclusive time (maximum): %12.6f ms\n", dur_in_max);
  printf("Inclusive time (average): %12.6f ms\n", dur_in_total/nruns);
  printf("Inclusive time (minimum): %12.6f ms\n", dur_in_min);

  return 0;
}
