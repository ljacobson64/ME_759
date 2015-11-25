#include "math.h"
#include "omp.h"
#include "stdio.h"
#include "time.h"

double f(double x) { return exp(sin(x)) * cos(x / 40.f); }

double integrate(double x_0, double h, int n) {
  int i;
  double result = 0.f;

#pragma omp parallel for reduction(+ : result)
  for (i = 4; i < n - 3; i++) result += f(i * h);
  
  result *= 48;

  result += 17 * (f(0 * h) + f((n - 0) * h));
  result += 59 * (f(1 * h) + f((n - 1) * h));
  result += 43 * (f(2 * h) + f((n - 2) * h));
  result += 49 * (f(3 * h) + f((n - 3) * h));

  result *= h / 48.f;
  return result;
}

int main() {
  double x_0 = 0;
  double x_n = 100;
  int n = 1e6;
  double h = (x_n - x_0) / n;

  int nthreads[2] = {1, 8};
  int nts = sizeof(nthreads) / sizeof(nthreads[0]);
  
  int i;
  for (i = 0; i < nts; i++) {
    int nt = nthreads[i];
    omp_set_num_threads(nt);

    int nruns = 0;
    double dur_total = 0.f;
    double dur, result;
    while (dur_total < 10.f) {
      nruns++;
      double start = omp_get_wtime();
      result = integrate(x_0, h, n);
      double end = omp_get_wtime();
      dur = end - start;
      dur_total += dur;
    }
    dur = dur_total / nruns;

    printf("threads: %d\n", nt);
    printf("result: %f\n", result);
    printf("time: %f\n", dur);
    printf("nruns: %d\n", nruns);
    printf("\n");
  }

  return 0;
}
