#include "math.h"
#include "omp.h"
#include "stdio.h"

double f(double x) { return exp(sin(x)) * cos(x / 40.f); }

double integrate(double x_0, double h, int n) {
  int i;
  double x;
  double result = 0.f;

#pragma omp parallel for reduction(+ : result)
  for (i = 4; i < n - 3; i++) {
    printf("thread %d\n", omp_get_thread_num());
    x = i * h;
    result += 48 * f(x);
  }

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

  omp_set_num_threads(8);

  double result = integrate(x_0, h, n);
  printf("result = %16.14f\n", result);

  return 0;
}
