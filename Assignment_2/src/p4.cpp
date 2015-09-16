#include <chrono>
#include <fstream>
#include <string>
#include <vector>

using std::ifstream;
using std::getline;
using std::string;
using std::vector;

// Function that counts the number of lines in a text file
int count_lines(char *fname) {
  ifstream finp(fname);
  string line;
  int num_lines = 0;
  
  while (getline(finp, line)) {
    num_lines++;
  }
  
  return num_lines;
}

// Function that reads a list of integers from file and returns them in vector
// form
vector<int> get_ints(char *fname, int num_ints) {
  ifstream finp(fname);
  vector<int> A(num_ints);
  int a;
  int i = 0;
  
  for (int i = 0; finp >> a; i++) {
    A[i] = a;
  }
  
  return A;
}

// Function that performs an exclusive scan on a vector
int exclusiveScan(vector<int> &A, int l) {
  
  int result = 0;
  
  for (int i = 0; i < l-1; i++) {
    result += A[i];
  }
  
  return result;
}

// Driver function
int main(int argc, char *argv[]) {
  // Count the number of lines in the text file
  int num_ints = count_lines(argv[1]);
  
  // Get the integers from the file listed on the command line
  vector<int> intVec = get_ints(argv[1], num_ints);
  int last_int = intVec[num_ints-1];
  
  // Run the exclusive scan algorithm
  auto start = std::chrono::high_resolution_clock::now();
  int result = exclusiveScan(intVec, num_ints);
  auto end = std::chrono::high_resolution_clock::now();
  long duration = std::chrono::  // microseconds
      duration_cast<std::chrono::microseconds>(end - start).count();
  int num_runs = 1;
  int i;
  
  // If the duration was found to be less than 1 second, double the number of
  // runs and try again
  while (duration < 1000000) {
    num_runs *= 2;
    
    start = std::chrono::high_resolution_clock::now();
    for (i = 0; i < num_runs; i++) {
      result = exclusiveScan(intVec, num_ints);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::
        duration_cast<std::chrono::microseconds>(end - start).count();
  }
  
  // Calculate the average duration in milliseconds
  double duration_avg = ((double)duration/(double)num_runs)/1000;
  
  // Print some information
  printf("Number of integers:    %12d\n", num_ints);
  printf("Last integer:          %12d\n", last_int);
  printf("Exclusive scan result: %12d\n", result);
  printf("Number of runs:        %12d\n", num_runs);
  printf("Time taken:            %12.6e ms\n", duration_avg);
  printf("\n");
  
  return 0;
}
