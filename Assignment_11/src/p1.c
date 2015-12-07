#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"

#define N_MAX 1073741824
#define MAX_RUNS 100000
#define DUR_MAX 1.

int main(int argc, char **argv) {
  int rank, procs;
  int tag = 0;
  unsigned int N;
  double start, end, dur, dur_total;
  int nruns;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Status status;

  int *data;
  data = (int *)malloc(N_MAX);

  // MPI_Bcast method
  if (rank == 0) {
    printf("____________MPI_Bcast method____________\n");
    printf("%12s  %12s  %12s\n", "Num_bytes", "Duration", "Num_runs");
  }

  for (N = 1; N <= N_MAX; N <<= 1) {
    dur_total = 0.;
    nruns = 0;
    while ((dur_total < DUR_MAX) && (nruns < MAX_RUNS)) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0) start = MPI_Wtime();
      MPI_Bcast(data, N, MPI_BYTE, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        end = MPI_Wtime();
        dur = end - start;
        dur_total += dur;
      }
      nruns++;
      MPI_Bcast(&dur_total, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
      dur = dur_total / nruns;
      printf("%12d  %12.8f  %12d\n", N, dur, nruns);
    }
  }

  // MPI_Send/MPI_Recv method
  if (rank == 0) {
    printf("\n________MPI_Send/MPI_Recv method____\n");
    printf("%12s  %12s  %12s\n", "Num_bytes", "Duration", "Num_runs");
  }

  for (N = 1; N <= N_MAX; N <<= 1) {
    dur_total = 0.;
    nruns = 0;
    while ((dur_total < DUR_MAX) && (nruns < MAX_RUNS)) {
      if (rank == 0) {
        int i;
        start = MPI_Wtime();
        for (i = 1; i < procs; i++)
          MPI_Send(data, N, MPI_BYTE, i, tag, MPI_COMM_WORLD);
      } else
        MPI_Recv(data, N, MPI_BYTE, 0, tag, MPI_COMM_WORLD, &status);
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0) {
        end = MPI_Wtime();
        dur = end - start;
        dur_total += dur;
      }
      nruns++;
      MPI_Bcast(&dur_total, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
      dur = dur_total / nruns;
      printf("%12d  %12.8f  %12d\n", N, dur, nruns);
    }
  }

  MPI_Finalize();
  return 0;
}
