#include "cuda.h"
#include "stdio.h"

int main(int argc, char *argv[]) {
  int version, log2N_min, log2N_max;
  float dur_max;
  if (argc == 5) {
    version = atoi(argv[1]);
    log2N_min = atoi(argv[2]);
    log2N_max = atoi(argv[3]);
    dur_max = atof(argv[4]) * 1000.f;
  } else {
    printf("Usage: ./p1 <version> <log2N_min> <log2N_max> <time>\n");
    printf("Version 0: Copy non-pinned memory from host to device\n");
    printf("Version 1: Copy pinned memory from host to device\n");
    printf(
        "Version 2: Copy non-pinned memory from host to device and back "
        "again\n");
    printf(
        "Version 3: Copy pinned memory from host to device and back again\n");
    printf("Time in ms\n");
    return 0;
  }
  int N_min = 1 << log2N_min;
  int N_max = 1 << log2N_max;
  float dur, dur_total;
  int num_runs;

  // Allocate host resources
  int *h;
  if (version == 0 || version == 2)
    h = (int *)malloc(N_max);
  else if (version == 1 || version == 3)
    cudaMallocHost(&h, N_max);

  // Allocate device resources
  int *d1, *d2;
  cudaMalloc(&d1, N_max);
  if (version == 2 || version == 3) cudaMalloc(&d2, N_max);

  printf("%8s %8s %12s %8s %12s\n", "Version", "log2N", "Bytes", "Runs",
         "Time");
  int log2N = log2N_min;
  int N = N_min;

  while (log2N <= log2N_max) {
    dur_total = 0.f;
    num_runs = 0;

    while (dur_total < dur_max) {
      num_runs++;

      // Setup timing
      cudaEvent_t start, end;
      cudaEventCreate(&start);
      cudaEventCreate(&end);

      if (version == 0 || version == 1) {
        // Copy to device
        cudaEventRecord(start, 0);
        cudaMemcpy(d1, h, N, cudaMemcpyHostToDevice);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
      } else if (version == 2 || version == 3) {
        // Copy to device and back again
        cudaEventRecord(start, 0);
        cudaMemcpy(d1, h, N, cudaMemcpyHostToDevice);
        cudaMemcpy(h, d2, N, cudaMemcpyDeviceToHost);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
      }

      // Calculate duration
      cudaEventElapsedTime(&dur, start, end);
      dur_total += dur;
    }

    dur = dur_total / num_runs;
    printf("%8d %8d %12d %8d %12.6f\n", version, log2N, N, num_runs, dur);

    log2N++;
    N *= 2;
  }

  printf("\n");

  // Free resources
  if (version == 0 || version == 2)
    free(h);
  else if (version == 1 || version == 3)
    cudaFree(h);
  cudaFree(d1);

  return 0;
}
