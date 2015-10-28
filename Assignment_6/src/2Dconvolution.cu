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

#include "cuda.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <fstream>

#include "2Dconvolution_gold.cpp"
#include "2Dconvolution.h"

using namespace std;

extern "C" void computeGold(float*, const float*, const float*, unsigned int,
                            unsigned int);
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
int CompareResults(float* A, float* B, int elements, float eps);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
bool ReadParams(int* params, int size, char* file_name);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void PrintMatrix(Matrix M);
void PrintMatrixDifs(Matrix M, Matrix N, float eps);
int int_power(int x, int n);

#define KR KERNEL_RADIUS
#define tx threadIdx.x
#define ty threadIdx.y

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P) {
  __shared__ float sM[KERNEL_SIZE][KERNEL_SIZE];
  __shared__ float sN[BLOCK_SIZE + 2 * KR][BLOCK_SIZE + 2 * KR];
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;

  // Load convolution kernel matirx into shared memory
  if (ty < KERNEL_SIZE && tx < KERNEL_SIZE)
    sM[ty][tx] = M.elements[ty * M.width + tx];

  // Load appropriate part of image matrix into shared memory
  // I have the strange sense that this is not optimized
  sN[ty + KR][tx + KR] = N.elements[(row)*N.width + col];
  if (tx < KR)
    if (col >= KR)
      sN[ty + KR][tx] = N.elements[(row)*N.width + col - KR];
    else
      sN[ty + KR][tx] = 0.f;
  if (tx >= BLOCK_SIZE - KR)
    if (col < N.width - KR)
      sN[ty + KR][tx + KR * 2] = N.elements[(row)*N.width + col + KR];
    else
      sN[ty + KR][tx + KR * 2] = 0.f;
  if (ty < KR)
    if (row >= KR)
      sN[ty][tx + KR] = N.elements[(row - KR) * N.width + col];
    else
      sN[ty][tx + KR] = 0.f;
  if (ty >= BLOCK_SIZE - KR)
    if (row < N.height - KR)
      sN[ty + KR * 2][tx + KR] = N.elements[(row + KR) * N.width + col];
    else
      sN[ty + KR * 2][tx + KR] = 0.f;
  if (ty < KR && tx < KR)
    if (row >= KR && col >= KR)
      sN[ty][tx] = N.elements[(row - KR) * N.width + col - KR];
    else
      sN[ty][tx] = 0.f;
  if (ty < KR && tx >= BLOCK_SIZE - KR)
    if (row >= KR && col < N.width - KR)
      sN[ty][tx + KR * 2] = N.elements[(row - KR) * N.width + col + KR];
    else
      sN[ty][tx + KR * 2] = 0.f;
  if (ty >= BLOCK_SIZE - KR && tx < KR)
    if (row < N.height - KR && col >= KR)
      sN[ty + KR * 2][tx] = N.elements[(row + KR) * N.width + col - KR];
    else
      sN[ty + KR * 2][tx] = 0.f;
  if (ty >= BLOCK_SIZE - KR && tx >= BLOCK_SIZE - KR)
    if (row < N.height - KR && col < N.width - KR)
      sN[ty + KR * 2][tx + KR * 2] =
          N.elements[(row + KR) * N.width + col + KR];
    else
      sN[ty + KR * 2][tx + KR * 2] = 0.f;

  // Make sure everything is loaded into shared memory
  __syncthreads();

  // Calculate the result
  float result = 0.f;
  for (int j = 0; j < KERNEL_SIZE; j++)
    for (int i = 0; i < KERNEL_SIZE; i++)
      result += sM[j][i] * sN[ty + j][tx + i];

  // Fill P matrix with result
  P.elements[row * N.width + col] = result;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
  Matrix M, N, P;
  bool compare = true;
  srand(2013);

  if (argc == 1 || argc == 2) {
    // Allocate and initialize the matrices
    M = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
    N = AllocateMatrix((rand() % 1024) + 1, (rand() % 1024) + 1, 0);
    P = AllocateMatrix(N.height, N.width, 1);
  } else if (argc == 3) {
    if (atoi(argv[1]) == 0) compare = false;
    int siz = int_power(2, atoi(argv[2]));
    M = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
    N = AllocateMatrix(siz, siz, 0);
    P = AllocateMatrix(siz, siz, 1);
  } else if (argc == 4 || argc == 5) {
    // Allocate and read in matrices from disk
    int* params = (int*)malloc(2 * sizeof(int));
    unsigned int data_read = 2;
    if (ReadParams(params, data_read, argv[1])) {
      printf("Error reading parameter file\n");
      return 1;
    }
    M = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
    N = AllocateMatrix(params[0], params[1], 0);
    P = AllocateMatrix(params[0], params[1], 0);
    (void)ReadFile(&M, argv[2]);
    (void)ReadFile(&N, argv[3]);
  }

  printf("Kernel size: %dx%d\n", KERNEL_SIZE, KERNEL_SIZE);
  printf("Block size:  %dx%d\n", BLOCK_SIZE, BLOCK_SIZE);
  printf("Image size:  %dx%d\n", N.height, N.width);

  // Convolution on the device
  ConvolutionOnDevice(M, N, P);

  // Compute the matrix convolution on the CPU for comparison
  Matrix reference = AllocateMatrix(P.height, P.width, 0);
  computeGold(reference.elements, M.elements, N.elements, N.height, N.width);

  // Check if the result is equivalent to the expected soluion
  if (compare) {
    int nDiffs = CompareResults(reference.elements, P.elements,
                                P.width * P.height, 0.001f);
    if (nDiffs == 0)
      printf("Looks good.\n");
    else
      printf("Doesn't look good: %d/%d are different\n", nDiffs,
             P.width * P.height);
  }

  if (argc == 5)
    WriteFile(P, argv[4]);
  else if (argc == 2)
    WriteFile(P, argv[1]);

  // Free matrices
  FreeMatrix(&M);
  FreeMatrix(&N);
  FreeMatrix(&P);

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P) {
  // Setup timing
  float dur_ex, dur_in;
  cudaEvent_t start_ex, end_ex, start_in, end_in;
  cudaEventCreate(&start_ex);
  cudaEventCreate(&end_ex);
  cudaEventCreate(&start_in);
  cudaEventCreate(&end_in);

  // Allocate device matrices
  Matrix Md = AllocateDeviceMatrix(M);
  Matrix Nd = AllocateDeviceMatrix(N);
  Matrix Pd = AllocateDeviceMatrix(P);

  // Setup the execution configuration
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((P.width + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (P.height + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Start inclusive timing
  cudaEventRecord(start_in, 0);

  // Load M and N to the device
  CopyToDeviceMatrix(Md, M);
  CopyToDeviceMatrix(Nd, N);

  // Start exclusive timing
  cudaEventRecord(start_ex, 0);

  // Launch the device computation threads
  ConvolutionKernel <<<dimGrid, dimBlock>>> (Md, Nd, Pd);

  // End exclusive timing
  cudaEventRecord(end_ex, 0);
  cudaEventSynchronize(end_ex);

  // Read P from the device
  CopyFromDeviceMatrix(P, Pd);

  // End inclusive timing
  cudaEventRecord(end_in, 0);
  cudaEventSynchronize(end_in);

  // Calculate durations
  cudaEventElapsedTime(&dur_ex, start_ex, end_ex);
  cudaEventElapsedTime(&dur_in, start_in, end_in);
  printf("GPU execution time (exclusive): %15.6f ms\n", dur_ex);
  printf("GPU execution time (inclusive): %15.6f ms\n", dur_in);
  cudaEventDestroy(start_ex);
  cudaEventDestroy(end_ex);
  cudaEventDestroy(start_in);
  cudaEventDestroy(end_in);

  // Free device matrices
  FreeDeviceMatrix(&Md);
  FreeDeviceMatrix(&Nd);
  FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M) {
  Matrix Mdevice = M;
  int size = M.width * M.height * sizeof(float);
  cudaMalloc((void**)&Mdevice.elements, size);
  return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, perform random initialization
//	If init == 1, allocate memory, but do not perform random initialization
//  If init == 2, initialize matrix parameters, but do not allocate memory
Matrix AllocateMatrix(int height, int width, int init) {
  Matrix M;
  M.width = M.pitch = width;
  M.height = height;
  M.elements = NULL;
  int size = M.width * M.height;

  // don't allocate memory on option 2
  if (init == 2) return M;

  M.elements = (float*)malloc(size * sizeof(float));

  // Don't fill with random numbers on option 1
  if (init == 1) return M;

  for (unsigned int i = 0; i < M.height * M.width; i++)
    M.elements[i] = 2.f * (rand() / (float)RAND_MAX) - 1.f;

  return M;
}

// Compare the data stored in two arrays on the host
int CompareResults(float* A, float* B, int elements, float eps) {
  int nDiffs = 0;
  for (unsigned int i = 0; i < elements; i++) {
    float error = abs(A[i] - B[i]);
    if (error > eps) nDiffs++;
  }
  return nDiffs;
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost) {
  int size = Mhost.width * Mhost.height * sizeof(float);
  Mdevice.height = Mhost.height;
  Mdevice.width = Mhost.width;
  Mdevice.pitch = Mhost.pitch;
  cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice) {
  int size = Mdevice.width * Mdevice.height * sizeof(float);
  cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M) {
  cudaFree(M->elements);
  M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M) {
  free(M->elements);
  M->elements = NULL;
}

// Read parameters from file
bool ReadParams(int* params, int size, char* file_name) {
  ifstream ifile(file_name);
  int i = 0;
  for (int i = 0; i < size; i++)
    if (ifile.fail() == false) ifile >> params[i];
  return (i == size) ? 1 : 0;
}

// Read a floating point matrix in from file
int ReadFile(Matrix* M, char* file_name) {
  unsigned int data_read = M->height * M->width;
  std::ifstream ifile(file_name);

  for (unsigned int i = 0; i < data_read; i++) ifile >> M->elements[i];
  ifile.close();
  return data_read;
}

// Write a floating point matrix to file
void WriteFile(Matrix M, char* file_name) {
  std::ofstream ofile(file_name);
  for (unsigned int i = 0; i < M.width * M.height; i++) ofile << M.elements[i];
  ofile.close();
}

// Print matrix to stdout
void PrintMatrix(Matrix M) {
  for (int j = 0; j < M.height; j++) {
    for (int i = 0; i < M.width; i++) {
      printf(" %5.2f", M.elements[j * M.width + i]);
    }
    printf("\n");
  }
  printf("\n");
}

// Print differences in matrices to stdout
void PrintMatrixDifs(Matrix M, Matrix N, float eps) {
  for (int j = 0; j < M.height; j++) {
    for (int i = 0; i < M.width; i++) {
      int ind = j * M.width + i;
      float error = abs(M.elements[ind] - N.elements[ind]);
      if (error < eps)
        printf(" _");
      else
        printf(" 1");
    }
    printf("\n");
  }
  printf("\n");
}

// Take integer power
int int_power(int x, int n) {
  if (x == 0) return 0;
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
