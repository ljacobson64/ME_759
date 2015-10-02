#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void sumArrays(double* dA, double* dB, double* dC) {
  int ind = blockDim.x*blockIdx.x + threadIdx.x;
  dC[ind] = dA[ind] + dB[ind];
}

int int_power(int base, int exponent) {
  int result = 1;
  for (int i = 0; i < exponent; i++) result *= base;
  return result;
}

double randBetween(int low, int high) {
  double result = (double)rand()/(double)RAND_MAX*(high - low) + low;
  return result;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: ./p2 log2(N) nthreads\n");
    exit(1);
  }

  int exponent = atoi(argv[1]);
  int N = int_power(2, exponent);  // Number of random numbers

  int nthreads = atoi(argv[2]);    // Number of threads per block
  int nblocks = N/nthreads;        // Number of blocks

  // Allocate host arrays
  int bytes = sizeof(double)*N;
  double *hA, *hB, *hC, *refC, *difC;
  hA   = (double*)malloc(bytes);
  hB   = (double*)malloc(bytes);
  hC   = (double*)malloc(bytes);
  refC = (double*)malloc(bytes);
  difC = (double*)malloc(bytes);

  // Allocate device arrays
  double *dA, *dB, *dC;
  cudaMalloc(&dA, bytes);
  cudaMalloc(&dB, bytes);
  cudaMalloc(&dC, bytes);

  // Fill host arrays with random numbers between -10 and 10 and sum them for
  // reference
  srand(1443740650);
  for (int i = 0; i < N; i++) {
    hA[i] = randBetween(-10, 10);
    hB[i] = randBetween(-10, 10);
    refC[i] = hA[i] + hB[i];
  }

  // Set up timing
  struct timespec start_in, end_in;
  float duration_ex;
  double duration_in, duration_ex_total;
  double duration_in_total = 0.0;
  long duration_in_ns;
  int num_runs = 0;

  // Loop until the total inclusive duration exceeds 1 second
  while (duration_in_total < 1000.0) {
    duration_ex_total = 0.0;
    duration_in_total = 0.0;
    
    // Double the number of runs
    if (num_runs != 0) num_runs *= 2;
    else num_runs = 1;
    
    for (int i = 0; i < num_runs; i++) {
      // CUDA timing variables
      cudaEvent_t start_ex, end_ex;
      cudaEventCreate(&start_ex);
      cudaEventCreate(&end_ex);
      
      // Start inclusive timing
      clock_gettime(CLOCK_MONOTONIC, &start_in);

      // Copy host arrays to the device
      cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
      cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

      // Start exclusive timing
      cudaEventRecord(start_ex, 0);

      // Invoke the device kernel which sums the arrays
      sumArrays<<<nblocks, nthreads>>>(dA, dB, dC);

      // End exclusive timing
      cudaEventRecord(end_ex, 0);
      cudaEventSynchronize(end_ex);

      // Copy the sum array back to the host
      cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

      // End inclusive timing
      clock_gettime(CLOCK_MONOTONIC, &end_in);

      // Calculate durations
      cudaEventElapsedTime(&duration_ex, start_ex, end_ex);
      cudaEventDestroy(start_ex);
      cudaEventDestroy(end_ex);
      duration_in_ns = (end_in.tv_sec - start_in.tv_sec)*1000000000L +
                        end_in.tv_nsec - start_in.tv_nsec;
      duration_in = (double)(duration_in_ns/1000000.0);
      duration_ex_total += (double)duration_ex;
      duration_in_total += duration_in;
    }
  }
  
  // Calculate average durations over all runs
  duration_ex = duration_ex_total/num_runs;
  duration_in = duration_in_total/num_runs;
  
  // Calculate the difference between the sum arrays and find the maximum
  // absolute difference
  double max_dif = 0.0;
  for (int i = 0; i < N; i++) {
    difC[i] = hC[i] - refC[i];
    if (abs(difC[i]) > max_dif) max_dif = abs(difC[i]);
  }

  // Free memory
  free(hA);
  free(hB);
  free(hC);
  free(refC);
  free(difC);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  // Print some information
  printf("Number of integers:  %12d\n", N);
  printf("Threads per block:   %12d\n", nthreads);
  printf("Number of runs:      %12d\n", num_runs);
  printf("Maximum difference:  %12.6e\n", max_dif);
  printf("Exclusive time:      %12.6e ms\n", duration_ex);
  printf("Inclusive time:      %12.6e ms\n", duration_in);
  printf("\n");

  return 0;
}
