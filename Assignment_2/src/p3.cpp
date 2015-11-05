#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

using std::ifstream;
using std::getline;
using std::stable_sort;  // Only used for comparison
using std::string;
using std::vector;

// Function that counts the number of lines in a text file
int count_lines(char *fname) {
  ifstream finp(fname);
  string line;
  int num_lines = 0;

  while (getline(finp, line)) num_lines++;

  return num_lines;
}

// Function that reads a list of integers from file and returns them in vector
// form
vector<int> get_ints(char *fname, int num_ints) {
  ifstream finp(fname);
  vector<int> A(num_ints);
  int a;
  int i = 0;

  for (int i = 0; finp >> a; i++) A[i] = a;

  return A;
}

// Function that finds the size, minimum, and maximum of a vector
void minmax(vector<int> &A, int &min_int, int &max_int) {
  int num_ints = A.size();

  // Special case where no array is passed
  if (num_ints <= 0) {
    min_int = 0;
    max_int = 0;
    return;
  }

  // Initially set min and max to the first value in the array
  min_int = A[0];
  max_int = A[0];

  // Check each value in the array to see if it is less than the stored min or
  // greater than the stored max and update accordingly
  for (int i = 1; i < num_ints; i++) {
    if (A[i] < min_int) min_int = A[i];
    if (A[i] > max_int) max_int = A[i];
  }

  return;
}

// Function that merges two pre-sorted halves of a specific part of a vector
// in a manner such that the final vector is sorted
void merge(vector<int> &A, vector<int> &B, int l, int m, int r) {
  // A[l..m] and A[m+1..r] are known to be sorted. They are merged by copying
  // them into another vector B in such a way that B[l..r] is sorted. At any
  // point, the next lowest value to add to B is the leftmost remaining value of
  // the left half of A (A[i]) or the leftmost remaining value of the right half
  // of A (A[j]).
  int i = l;      // Index for left half
  int j = m + 1;  // Index for right half
  int k = l;      // Index for whole vector
  while (i <= m && j <= r) {
    if (A[i] <= A[j]) {
      B[k] = A[i];
      i++;
    } else {
      B[k] = A[j];
      j++;
    }
    k++;
  }

  // After all of one of the halves of A has been completely copied into B,
  // there will be some values left over in the other half. Append these to the
  // end of B.
  while (i <= m) {
    B[k] = A[i];
    i++;
    k++;
  }
  while (j <= r) {
    B[k] = A[j];
    j++;
    k++;
  }

  // Copy the relevant sorted part of B back into A
  for (k = l; k <= r; k++) A[k] = B[k];

  return;
}

// Recursive function that sorts a vector of integers
void mergeSort(vector<int> &A, vector<int> &B, int l, int r) {
  if (l >= r) return;

  int m = l + (r - l) / 2;    // (l+r)/2 overflows if l+r > INT_MAX
  mergeSort(A, B, l, m);      // Sort the first and second "quarters" of A
  mergeSort(A, B, m + 1, r);  // Sort the third and fourth "quarters" of A
  merge(A, B, l, m, r);       // Merge the two sorted halves of A

  return;
}

// Run and time the sorting algorithm
double SortAndTime(int sortType, vector<int> intVec, int &num_runs) {
  // Allocate the main vector A and work vector B
  int num_ints = intVec.size();
  vector<int> A = intVec;
  vector<int> B(num_ints);

  // Declare some variables
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  int i;
  num_runs = 1;

  // Run and time a single run of the the merge sort algorithm
  if (sortType == 0) {  // My merge sort algorithm
    start = std::chrono::high_resolution_clock::now();
    mergeSort(A, B, 0, num_ints - 1);
    end = std::chrono::high_resolution_clock::now();
  } else if (sortType == 1) {  // Standard library stable sort algorithm
    start = std::chrono::high_resolution_clock::now();
    stable_sort(A.begin(), A.end());
    end = std::chrono::high_resolution_clock::now();
  } else
    return (double)0.0;

  long duration = std::chrono::  // microseconds
                  duration_cast<std::chrono::microseconds>(end - start).count();

  // If the duration was found to be less than 1 second, double the number of
  // runs and try again
  while (duration < 1000000) {
    num_runs *= 2;

    if (sortType == 0) {  // My merge sort algorithm
      start = std::chrono::high_resolution_clock::now();
      for (i = 0; i < num_runs; i++) {
        A = intVec;
        mergeSort(A, B, 0, num_ints - 1);
      }
      end = std::chrono::high_resolution_clock::now();
    } else if (sortType == 1) {  // Standard library stable sort algorithm
      start = std::chrono::high_resolution_clock::now();
      for (i = 0; i < num_runs; i++) {
        A = intVec;
        stable_sort(A.begin(), A.end());
      }
      end = std::chrono::high_resolution_clock::now();
    }

    duration = std::chrono::duration_cast<std::chrono::microseconds>(
                   end - start).count();

    // Since the vector A must be refilled with the original values each time,
    // this amount of time must be subtracted out of the final result
    start = std::chrono::high_resolution_clock::now();
    for (i = 0; i < num_runs; i++) A = intVec;
    end = std::chrono::high_resolution_clock::now();
    duration -= std::chrono::duration_cast<std::chrono::microseconds>(
                    end - start).count();
  }

  // Calculate the average duration in milliseconds
  double duration_avg = ((double)duration / (double)num_runs) / 1000;

  return duration_avg;
}

// Driver function
int main(int argc, char *argv[]) {
  // Count the number of lines in the text file
  int num_ints = count_lines(argv[1]);

  // Get the integers from the file listed on the command line
  vector<int> intVec = get_ints(argv[1], num_ints);

  // Get the min and max
  int min_int, max_int;
  minmax(intVec, min_int, max_int);

  // Time my merge sort algorithm
  int num_runs_lucas;
  double duration_lucas = SortAndTime(0, intVec, num_runs_lucas);

  // Time the standard library stable sort algorithm
  int num_runs_std;
  double duration_std = SortAndTime(1, intVec, num_runs_std);

  // Print some information
  printf("Number of integers:           %12d\n", num_ints);
  printf("Minimum integer:              %12d\n", min_int);
  printf("Maximum integer:              %12d\n", max_int);
  printf("Number of runs (Lucas):       %12d\n", num_runs_lucas);
  printf("Time taken (Lucas):           %12.6e ms\n", duration_lucas);
  printf("Number of runs (stable_sort): %12d\n", num_runs_std);
  printf("Time taken (stable_sort):     %12.6e ms\n", duration_std);
  printf("\n");

  return 0;
}
