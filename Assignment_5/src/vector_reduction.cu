/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

#ifdef _WIN32
#define NOMINMAX
#endif

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Includes
#include "vector_reduction_kernel.cu"
#include "vector_reduction_gold.cpp"

////////////////////////////////////////////////////////////////////////////////
// Declarations, forward
////////////////////////////////////////////////////////////////////////////////
int runTest(int argc, char** argv);
int ReadFile(float* M, char* file_name);
float computeOnDevice(float* h_data, int num_elements, int block_size);
int int_power(int x, int n);

extern "C" void computeGold(float* reference, float* idata,
                            const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
  runTest(argc, argv);
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Run test
////////////////////////////////////////////////////////////////////////////////
int runTest(int argc, char** argv) {
  int num_elements, block_size;
  if (argc == 3) {
    num_elements = int_power(2, atoi(argv[1]));
    block_size = int_power(2, atoi(argv[2]));
  } else {
    num_elements = 1024;
    block_size = 1024;
  }

  int errorM = 0;

  const unsigned int array_mem_size = sizeof(float) * num_elements;

  // Allocate host memory to store the input data
  float* h_data = (float*)malloc(array_mem_size);

  // * No arguments: Randomly generate input data and compare against the host's
  //   result.
  // * One argument: Read the input data array from the given file.
  switch (argc - 1) {
    case 1:  // One Argument
      errorM = ReadFile(h_data, argv[1]);
      if (errorM != 1) {
        printf("Error reading input file!\n");
        exit(1);
      }
      break;

    default:  // No Arguments or one argument
      // Initialize the input data on the host to be integer values between 0
      // and 1000
      for (unsigned int i = 0; i < num_elements; ++i) {
        h_data[i] = floorf(1000 * (rand() / (float)RAND_MAX));
      }
      break;
  }

  // Compute reference solution
  float reference = 0.0f;
  computeGold(&reference, h_data, num_elements);

  // Compute solution on GPU
  float result = computeOnDevice(h_data, num_elements, block_size);

  // Run accuracy test
  float epsilon = 0.0001f;
  unsigned int result_regtest = (abs(result / reference - 1.f) <= epsilon);
  printf("Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
  printf("num_elements: %d\n", num_elements);
  printf("block_size:   %d\n", block_size);
  printf("num_blocks:   %d\n", num_elements / block_size);
  printf("GPU result:   %f\n", result);
  printf("CPU result:   %f\n", reference);
  printf("\n");

  // Cleanup memory
  free(h_data);

  return 0;
}

int ReadFile(float* M, char* file_name) {
  // unsigned int elements_read = NUM_ELEMENTS;
  // if (cutReadFilef(file_name, &M, &elements_read, true))
  //   return 1;
  // else
  //   return 0;
  return 0;
}

// Take h_data from host, copies it to device, setup grid and thread dimensions,
// excutes kernel function, and copy result of scan back to h_data.
// Note: float* h_data is both the input and the output of this function.
float computeOnDevice(float* h_data, int num_elements, int block_size) {
  int num_blocks = num_elements / block_size;

  // Allocate device array
  float* g0_data, *g1_data, *g2_data;
  cudaMalloc(&g0_data, sizeof(float) * num_elements);
  cudaMalloc(&g1_data, sizeof(float) * num_blocks);
  cudaMalloc(&g2_data, sizeof(float));

  // Copy host array to device
  cudaMemcpy(g0_data, h_data, sizeof(float) * num_elements,
             cudaMemcpyHostToDevice);

  // Execute device kernel
  reduce0 <<<num_blocks, block_size, sizeof(float) * block_size>>>
      (g0_data, g1_data, num_elements);
  reduce0 <<<1, num_blocks, sizeof(float) * num_blocks>>>
      (g1_data, g2_data, num_blocks);

  // Copy the result array back to the host
  float* result = (float*)malloc(sizeof(float));
  cudaMemcpy(result, g2_data, sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup memory
  cudaFree(g0_data);
  cudaFree(g1_data);
  cudaFree(g2_data);

  return *result;
}

// Take integer power
int int_power(int x, int n) {
  if (n <= 0) return 1;
  int y = 1;
  while (n > 1) {
    if (n % 2 == 0) {
      x *= x;
      n /= 2;
    } else {
      y *= x;
      x *= x;
      n = (n - 1) / 2;
    }
  }
  return x * y;
}
