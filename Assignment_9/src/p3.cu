#include "stdio.h"
#include "stdlib.h"
#include <vector>

#include "thrust/device_vector.h"
#include "thrust/inner_product.h"

int compute_number_of_days_with_rainfall(thrust::device_vector<int>& day) {
  return thrust::inner_product(day.begin(), day.end() - 1, day.begin() + 1, 0,
                               thrust::plus<int>(),
                               thrust::not_equal_to<int>()) + 1;
}

int main(int argc, char** argv) {
  int N = 15;
  int h_day[] = {0, 0, 1, 2, 5, 5, 6, 6, 7, 8, 9, 9, 9, 10, 11};
  int h_site[] = {2, 3, 0, 1, 1, 2, 0, 1, 2, 1, 3, 4, 0, 1, 2};
  int h_measure[] = {9, 5, 6, 3, 3, 8, 2, 6, 5, 10, 9, 11, 8, 4, 1};

  thrust::device_vector<int> d_day(N), d_site(N), d_measure(N);

  for (int i = 0; i < N; i++) {
    d_day[i] = h_day[i];
    d_site[i] = h_site[i];
    d_measure[i] = h_measure[i];
  }

  int number_of_days_with_rainfall = compute_number_of_days_with_rainfall(d_day);
  printf("%d\n", number_of_days_with_rainfall);

  return 0;
}
