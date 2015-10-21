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

/* Matrix multiplication: C = A * B.
 * Host code.
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
using namespace std;

// Includes
#include "tiledMatMult.h"
#include "matrixmul_kernel.cuh"
#include "matrixmul_gold.cpp"

////////////////////////////////////////////////////////////////////////////////
// Declarations, forward
////////////////////////////////////////////////////////////////////////////////
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
Matrix PaddedMatrix(const Matrix& M, const int BLKSZ, int copyEntries);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void ExtractFromPadded(Matrix M, const Matrix& Mpadded);
bool CompareResults(float* A, float* B, int elements, float eps);
int ReadFile(Matrix* M, char* file_name);
bool ReadParams(int* params, int size, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
void PrintMatrix(Matrix M);

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);

extern "C" void computeGold(float*, const float*, const float*, unsigned int,
                            unsigned int, unsigned int);

#define MAT_MAX_SIZE 4096  // Max size that works on Euler is 11360,
                           // but this takes forever on the CPU

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
  Matrix M, N, P;
  int errorM = 0, errorN = 0;
  bool compare = true;

  if (argc == 1 || argc == 2) {
    // Allocate and initialize the matrices
    srand(52);
    int dummy;
    dummy = rand() % MAT_MAX_SIZE;
    int Mh = (dummy == 0 ? 1 : dummy);
    dummy = rand() % MAT_MAX_SIZE;
    int Mw = (dummy == 0 ? 1 : dummy);
    dummy = rand() % MAT_MAX_SIZE;
    int Nw = (dummy == 0 ? 1 : dummy);
    M = AllocateMatrix(Mh, Mw, 0);
    N = AllocateMatrix(Mw, Nw, 0);
    P = AllocateMatrix(Mh, Nw, 1);
  } else if (argc == 3) {
    if (atoi(argv[1]) == 0) compare = false;
    int siz = atoi(argv[2]);
    M = AllocateMatrix(siz, siz, 0);
    N = AllocateMatrix(siz, siz, 0);
    P = AllocateMatrix(siz, siz, 1);
  } else if (argc == 4 || argc == 5) {
    // Allocate and read in matrices from disk
    int* params = (int*)malloc(3 * sizeof(int));
    unsigned int data_read = 3;
    ReadParams(params, data_read, argv[1]);
    if (data_read != 3) {
      printf("Error reading parameter file\n");
      return 1;
    }

    M = AllocateMatrix(params[0], params[1], 0);
    N = AllocateMatrix(params[1], params[2], 0);
    P = AllocateMatrix(params[0], params[2], 1);
    errorM = ReadFile(&M, argv[2]);
    errorN = ReadFile(&N, argv[3]);
    if (errorM || errorN) {
      printf("Error reading input files %d, %d\n", errorM, errorN);
      return 1;
    }
  }

  printf("Block size: %d\n", BLOCK_SIZE);
  printf("Dimension M[height,width]: %d  %d\n", M.height, M.width);
  printf("Dimension N[height,width]: %d  %d\n", N.height, N.width);

  // M * N on the device
  MatrixMulOnDevice(M, N, P);

  if (compare) {
    // Setup CPU timing
    float dur_cpu;
    cudaEvent_t start_cpu, end_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&end_cpu);

    // Compute the matrix multiplication on the CPU for comparison
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
    cudaEventRecord(start_cpu, 0);
    computeGold(reference.elements, M.elements, N.elements, M.height, M.width,
                N.width);
    cudaEventRecord(end_cpu, 0);
    cudaEventSynchronize(end_cpu);

    // Calculate duration
    cudaEventElapsedTime(&dur_cpu, start_cpu, end_cpu);
    printf("CPU execution time:             %15.6f ms\n", dur_cpu);
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(end_cpu);

    // In this case check if the result is equivalent to the expected soluion
    bool res = CompareResults(reference.elements, P.elements,
                              P.height * P.width, 0.01f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
  }

  printf("\n");

  if (argc == 5) {
    WriteFile(P, argv[4]);
  } else if (argc == 2) {
    WriteFile(P, argv[1]);
  }

  // Free matrices
  FreeMatrix(&M);
  FreeMatrix(&N);
  FreeMatrix(&P);
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
//  Multiply on the device
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix Munpadded, const Matrix Nunpadded,
                       Matrix Punpadded) {
  // Setup timing
  float dur_ex, dur_in;
  cudaEvent_t start_ex, end_ex, start_in, end_in;
  cudaEventCreate(&start_ex);
  cudaEventCreate(&end_ex);
  cudaEventCreate(&start_in);
  cudaEventCreate(&end_in);

  // I'm going to take care of the padding here...
  Matrix M = PaddedMatrix(Munpadded, BLOCK_SIZE, 1);
  Matrix N = PaddedMatrix(Nunpadded, BLOCK_SIZE, 1);
  Matrix P = PaddedMatrix(Punpadded, BLOCK_SIZE, 0);

  // Allocate device matrices
  Matrix Md = AllocateDeviceMatrix(M);
  Matrix Nd = AllocateDeviceMatrix(N);
  Matrix Pd = AllocateDeviceMatrix(P);

  // Setup the execution configuration
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(P.width / BLOCK_SIZE, P.height / BLOCK_SIZE);

  // Start inclusive timing
  cudaEventRecord(start_in, 0);

  // Load M and N to the device
  CopyToDeviceMatrix(Md, M);
  CopyToDeviceMatrix(Nd, N);
  // CopyToDeviceMatrix(Pd, P);

  // Start exclusive timing
  cudaEventRecord(start_ex, 0);

  // Launch the device computation threads
  MatrixMulKernel << <dimGrid, dimBlock>>> (Md, Nd, Pd);

  // End exclusive timing
  cudaEventRecord(end_ex, 0);
  cudaEventSynchronize(end_ex);

  // Read P from the device
  CopyFromDeviceMatrix(P, Pd);

  // End inclusive timing
  cudaEventRecord(end_in, 0);
  cudaEventSynchronize(end_in);

  // Extract the submatrix with the result
  ExtractFromPadded(Punpadded, P);

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

  // Free the helper padded matrices
  FreeMatrix(&M);
  FreeMatrix(&N);
  FreeMatrix(&P);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M) {
  Matrix Mdevice = M;
  int size = M.width * M.height * sizeof(float);
  cudaMalloc((void**)&Mdevice.elements, size);
  return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//  If init == 0, initialize to all zeroes.
//  If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory
Matrix AllocateMatrix(int height, int width, int init) {
  Matrix M;
  M.width = M.pitch = width;
  M.height = height;
  int size = M.width * M.height;
  M.elements = NULL;

  // Don't allocate memory on option 2
  if (init == 2) return M;

  M.elements = (float*)malloc(size * sizeof(float));

  // Don't fill with random numbers on option 1
  if (init == 1) return M;

  for (unsigned int i = 0; i < M.height * M.width; i++) {
    M.elements[i] = (init == 0) ? (0.0f) : (rand() * 3 / (float)RAND_MAX);
  }
  return M;
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

// Compare the data stored in two arrays on the host
bool CompareResults(float* A, float* B, int elements, float eps) {
  for (unsigned int i = 0; i < elements; i++) {
    float error = fabs(A[i] - B[i]);
    if (error > eps) {
      return false;
    }
  }
  return true;
}

bool ReadParams(int* params, int size, char* file_name) {
  ifstream ifile(file_name);
  int i = 0;
  for (int i = 0; i < size; i++) {
    if (ifile.fail() == false) {
      ifile >> params[i];
    }
  }
  return (i == size) ? 1 : 0;
}

// Read a floating point matrix in from file
// Returns 0 if the number of elements read equals M.height * M.width,
// and 1 otherwise
int ReadFile(Matrix* M, char* file_name) {
  unsigned int data_read = M->height * M->width;
  std::ifstream ifile(file_name);
  unsigned int i = 0;
  for (; i < data_read; i++) {
    ifile >> M->elements[i];
  }
  ifile.close();
  return (i == data_read) ? 0 : 1;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name) {
  std::ofstream ofile(file_name);
  for (unsigned int i = 0; i < M.width * M.height; i++) {
    ofile << M.elements[i] << " ";
  }
  ofile.close();
}

// Given a matrix M, produce a padded matrix that has both dimensions a
// multiple of BLKSZ.  The elements of the original M matrix can be
// copied over to the new padded matrix provided the flag copyEntries
// is not zero.  Note that the assumption is that M.pitch <= M.width;
Matrix PaddedMatrix(const Matrix& M, const int BLKSZ, int copyEntries) {
  Matrix Mpadded;
  int dummy = (M.height - 1) / BLKSZ + 1;
  Mpadded.height = dummy * BLKSZ;
  dummy = (M.width - 1) / BLKSZ + 1;
  Mpadded.width = dummy * BLKSZ;
  Mpadded.pitch = M.width;
  Mpadded.elements =
      (float*)calloc(Mpadded.width * Mpadded.height, sizeof(float));

  // Copy entries of original matrix only if asked to
  if (copyEntries) {
    for (int i = 0; i < M.height; i++) {
      memcpy(&Mpadded.elements[i * Mpadded.width], &M.elements[i * M.width],
             M.width * sizeof(float));
    }
  }
  return Mpadded;
}

// The submatrix of dimensions M.width by M.height of Mpadded is copied over
// from Mpadded into M.  Note that the assumption is that M.pitch <= M.width;
void ExtractFromPadded(Matrix M, const Matrix& Mpadded) {
  if (Mpadded.pitch != M.width) {
    printf("Error extracting data from padded matrix: Number of rows %d, %d\n",
           Mpadded.pitch, M.width);
    exit(1);
  }

  if (Mpadded.height < M.height) {
    printf("Error extracting data from padded matrix: Height too small%d, %d\n",
           Mpadded.height, M.height);
    exit(1);
  }

  for (int i = 0; i < M.height; i++) {
    memcpy(&M.elements[i * M.width], &Mpadded.elements[i * Mpadded.width],
           M.width * sizeof(float));
  }

  return;
}

// Print the contents of a matrix to stdout
void PrintMatrix(Matrix M) {
  for (unsigned int i = 0; i < M.height; i++) {
    for (unsigned int j = 0; j < M.width; j++) {
      printf("%7.4f ", M.elements[i * M.width + j]);
    }
    printf("\n");
  }
  printf("\n");
}
