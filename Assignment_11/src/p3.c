#include "math.h"
#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define N 1000000
#define MAX_RUNS 100000
#define DUR_MAX 1.

double f(double x) { return exp(sin(x)) * cos(x / 40.f); }

int main(int argc, char **argv) {
  // Setup MPI run
  int rank, procs, i, p;
  int tag = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Status status;

  // Setup timing
  double start, end, dur, dur_total;
  int nruns = 0;

  // Setup grid
  double x_0 = 0;
  double x_n = 100;
  double h = (x_n - x_0) / N;
  double result;
  double result_p[procs];

  // Figure out which values will be computed on each core
  int nb[procs + 1];
  if (rank == 0) {
    int npc[procs];
    int nr = N - 7;
    int pr = procs;
    nb[0] = 0;
    for (p = 0; p < procs; p++) {
      npc[p] = (nr + pr - 1) / pr;
      nb[p + 1] = nb[p] + npc[p];
      nr -= npc[p];
      pr--;
    }
  }
  MPI_Bcast(nb, procs + 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Do multiple runs to get more accurate timing results
  while ((dur_total < DUR_MAX) && (nruns < MAX_RUNS)) {
    nruns++;

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) start = MPI_Wtime();

    // Compute part of the central sum on each core
    result_p[rank] = 0.;
    for (i = nb[rank]; i < nb[rank + 1]; i++) result_p[rank] += f(i * h);

    // Send the partial central sum from each core to core 0
    if (rank == 0)
      for (p = 1; p < procs; p++)
        MPI_Recv(&result_p[p], 1, MPI_DOUBLE, p, tag, MPI_COMM_WORLD, &status);
    else
      MPI_Send(&result_p[rank], 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);

    // Do some post-processing on core 0 (this is a much smaller amount of work
    // than the computation of the central sum)
    if (rank == 0) {
      result = 0.;
      for (p = 0; p < procs; p++)
        result += result_p[p];
      result *= 48;
      result += 17 * (f(0 * h) + f((N - 0) * h));
      result += 59 * (f(1 * h) + f((N - 1) * h));
      result += 43 * (f(2 * h) + f((N - 2) * h));
      result += 49 * (f(3 * h) + f((N - 3) * h));
      result *= h / 48.f;
    }

    // End timing
    if (rank == 0) {
      end = MPI_Wtime();
      dur = end - start;
      dur_total += dur;
    }
    MPI_Bcast(&dur_total, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // Print some stuff
  if (rank == 0) {
    dur = dur_total / nruns;
    printf("N: %d\n", N);
    printf("cores: %d\n", procs);
    printf("time: %f ms\n", dur);
    printf("num_runs: %d\n", nruns);
    printf("result: %f\n", result);
    printf("\n");
  }

  MPI_Finalize();
  return 0;
}
