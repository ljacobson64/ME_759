#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16

__global__ void cudaMultiplyArrays(int* dA, int* dB, int* dC,
                                   int hA, int wA, int wB, int wC) {
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  if (row > hA || col > wB) return;
  int result = 0;
  for (int e = 0; e < wA; e++)
    result += dA[row*wA + e]*dB[e*wB + col];
  dC[row*wC + col] = result;
}

void fill_array(int &A, int hA, int wA) {
  for (int i = 0; i < hA; i++)
    for (int j = 0; j < wA; j++)
      A[i*wA + j] = i + j;
}

int main() {
  // Array sizes
  int m = 16;
  int n = 32;
  int p =  1;
  int hA = m, wA = n;
  int hB = n, wB = p;
  int hC = m, wC = p;
  int sA = hA*wA;
  int sB = hB*wB;
  int sC = hC*wC;
  
  // Allocate host arrays
  int *A, *B, *C;
  A = (int*)malloc(sizeof(int)*sA);
  B = (int*)malloc(sizeof(int)*sB);
  C = (int*)malloc(sizeof(int)*sC);
  
  // Allocate device arrays
  int *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(int)*sA);
  cudaMalloc(&dB, sizeof(int)*sB);
  cudaMalloc(&dC, sizeof(int)*sC);
  
  // Fill A and B with some integers
  fill_array(A, hA, wA)
  fill_array(B, hB, wB)
  
  // Copy host arrays to the device
  cudaMemcpy(dA, A, sizeof(int)*sA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(int)*sB, cudaMemcpyHostToDevice);
  
  // Invoke the device kernel which multiplies the arrays
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((wC + BLOCK_SIZE - 1)/dimBlock.x, (hC + BLOCK_SIZE - 1)/dimBlock.y);
  cudaMultiplyArrays<<<dimGrid, dimBlock>>>(dA, dB, dC, hA, wA, wB, wC);
  
  // Copy the result array back to the host
  cudaMemcpy(C, dC, sizeof(int)*sC, cudaMemcpyDeviceToHost);
  
  // Write result to file
  //FILE *fp;
  //fp = fopen("problem2.out", "w");
  //for (int i = 0; i < hC; i++)
  //  for (int j = 0; j < wC; j++)
  //    fprintf(fp, "%12d\n", C[i*wC + j]);
  //fclose(fp);
  for (int i = 0; i < hC; i++) {
    for (int j = 0; j < wC; j++) {
      B[i*wC + j] = i + j;
      printf("%12d ", C[i*wC + j]);
    }
    printf("\n");
  }
  printf("\n");
  
  // Free memory
  free(A);
  free(B);
  free(C);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  
  return 0;
}
