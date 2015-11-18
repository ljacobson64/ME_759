#include <iostream>
#include <vector>

#include "thrust/count.h"
#include "thrust/device_vector.h"
#include "thrust/inner_product.h"
#include "thrust/sort.h"

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

void rainfall_per_site(const Data& data, thrust::device_vector<int>& site,
                       thrust::device_vector<int>& measure, int num_sites) {
  thrust::device_vector<int> tmp_site = data.site;
  thrust::device_vector<int> tmp_measure = data.measure;
  site.resize(num_sites);
  measure.resize(num_sites);
  thrust::sort_by_key(tmp_site.begin(), tmp_site.end(), tmp_measure.begin());
  thrust::reduce_by_key(tmp_site.begin(), tmp_site.end(), tmp_measure.begin(),
                        site.begin(), measure.begin());
}

int main(int argc, char** argv) {
  int h_day[] = {0, 0, 1, 2, 5, 5, 6, 6, 7, 8, 9, 9, 9, 10, 11};
  int h_site[] = {2, 3, 0, 1, 1, 2, 0, 1, 2, 1, 3, 4, 0, 1, 2};
  int h_measure[] = {9, 5, 6, 3, 3, 8, 2, 6, 5, 10, 9, 11, 8, 4, 1};
  int N = 15;
  int num_sites = 5;

  Data data;
  data.day.resize(N);
  data.site.resize(N);
  data.measure.resize(N);

  for (int i = 0; i < N; i++) {
    data.day[i] = h_day[i];
    data.site[i] = h_site[i];
    data.measure[i] = h_measure[i];
  }

  // Problem A
  int days_over_5 = days_over_threshold(data, 5);
  std::cout << "Number of days over 5: " << days_over_5 << std::endl;

  // Problem B
  thrust::device_vector<int> site, measure;
  rainfall_per_site(data, site, measure, num_sites);
  for (int i = 0; i < num_sites; i++)
    std::cout << "Site " << site[i] << ": " << measure[i] << std::endl;

  return 0;
}
