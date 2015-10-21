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
 * disclosure to any third parthreadIdx.y without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILIthreadIdx.y OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANthreadIdx.y OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILIthreadIdx.y, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
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
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"
#include "tiledMatMult.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionalithreadIdx.y
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(const Matrix M, const Matrix N, Matrix P) {
  // Thread and block indices
  unsigned int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  unsigned int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  // Initialize subarrays in shared memory
  __shared__ float sM[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sN[BLOCK_SIZE][BLOCK_SIZE];

  // Loop over each subarray
  float result = 0.f;
  for (unsigned short r = 0; r < M.width / BLOCK_SIZE; r++) {
    // Fill the subarrays in shared memory
    sM[threadIdx.y][threadIdx.x] = M.elements[col * M.width + (r * BLOCK_SIZE + threadIdx.x)];
    sN[threadIdx.y][threadIdx.x] = N.elements[(r * BLOCK_SIZE + threadIdx.y) * N.width + row];

    __syncthreads();

    // Sum the contributions from each thread in the subarray
    for (unsigned short s = 0; s < BLOCK_SIZE; s++) {
      result += sM[threadIdx.y][s] * sN[s][threadIdx.x];
    }

    __syncthreads();
  }

  // Fill result array
  P.elements[col * N.width + row] = result;
}

#endif  // #ifndef _MATRIXMUL_KERNEL_H_
