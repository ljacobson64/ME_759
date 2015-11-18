#include "stdio.h"
#include "stdlib.h"
#include <vector>

#include "thrust/count.h"
#include "thrust/device_vector.h"
#include "thrust/inner_product.h"

struct Data {
  thrust::device_vector<int> day;
  thrust::device_vector<int> site;
  thrust::device_vector<int> measure;
};

int days_with_rainfall(const Data& data) {
  return thrust::inner_product(data.day.begin(), data.day.end() - 1,
                               data.day.begin() + 1, 1, thrust::plus<int>(),
                               thrust::not_equal_to<int>());
}

int days_over_threshold(const Data& data, const int threshold) {
  int N = days_with_rainfall(data);
  thrust::device_vector<int> day(N), measure(N);
  thrust::reduce_by_key(data.day.begin(), data.day.end(), data.measure.begin(),
                        day.begin(), measure.begin());
  return thrust::count_if(measure.begin(), measure.end(),
                          thrust::placeholders::_1 > threshold);
}

int main(int argc, char** argv) {
  int N = 15;
  int h_day[] = {0, 0, 1, 2, 5, 5, 6, 6, 7, 8, 9, 9, 9, 10, 11};
  int h_site[] = {2, 3, 0, 1, 1, 2, 0, 1, 2, 1, 3, 4, 0, 1, 2};
  int h_measure[] = {9, 5, 6, 3, 3, 8, 2, 6, 5, 10, 9, 11, 8, 4, 1};

  Data data;
  data.day.resize(N);
  data.site.resize(N);
  data.measure.resize(N);

  for (int i = 0; i < N; i++) {
    data.day[i] = h_day[i];
    data.site[i] = h_site[i];
    data.measure[i] = h_measure[i];
  }

  int days_over_0 = days_with_rainfall(data);
  int days_over_5 = days_over_threshold(data, 5);
  printf("%d\n", days_over_0);
  printf("%d\n", days_over_5);

  return 0;
}
