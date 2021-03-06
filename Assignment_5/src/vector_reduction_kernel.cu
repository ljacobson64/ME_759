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

#ifndef _VECTOR_REDUCTION_KERNEL_H_
#define _VECTOR_REDUCTION_KERNEL_H_

////////////////////////////////////////////////////////////////////////////////
//! @param gi_data  input data in global memory
//! @param go_data  output data in global memory
//! @param n        input number of elements to scan from input data
////////////////////////////////////////////////////////////////////////////////
__global__ void reduction(float *gi_data, float *go_data, int n) {
  // Setup shared memory
  extern __shared__ float s_data[];

  // Load global memory into shared memory
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[threadIdx.x] = gi_data[i];

  // Make sure all the memory in a block is loaded before continuing
  __syncthreads();

  // Add the first and second halves of the array and place the result in the
  // first half. Then add the first and second halves of the original first
  // half, and repeat until the final block sum is computed. The total number of
  // loops is equal to log_2(blockDim.x).
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset
       s_data[threadIdx.x] += s_data[threadIdx.x + offset];
    __syncthreads();
  }

  // Write the result for each block into go_data
  if (threadIdx.x == 0) go_data[blockIdx.x] = s_data[0];
}

#endif  // #ifndef _VECTOR_REDUCTION_KERNEL_H_
