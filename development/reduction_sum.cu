#include <iostream>
#include <cassert>
#include <ctime>
#include <stdio.h>
#include <cuda.h>
/*********This is a workaround to the bug of Nvidia CUDA identified at this page***********/
/*https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error*/
__device__ double atomic_add(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Use integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

using namespace std;

#define n 131072 
#define d 512
#define WARP_SIZE 32

__global__ void parallel_sum(const double* __restrict__ z, double *x) {
  // Holds intermediates in shared memory reduction
  __syncthreads();
  __shared__ double buffer[1024/WARP_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % WARP_SIZE;

  for (int k=0; k<d; k++) {
    int j = (k + blockIdx.x) % d;
    double temp;
    // All threads in a block of 1024 take an element
    temp = z[i + n*j];
    
    // All warps in this block (32) compute the sum of all threads in
    // their warp
    for(int delta = WARP_SIZE/2; delta > 0; delta /= 2)
      temp += __shfl_xor_sync(0xffffffff, temp, delta);

    // Write all 32 of these partial sums to shared memory
    if(lane == 0)
      buffer[threadIdx.x / WARP_SIZE] = temp;
    
    __syncthreads();

    // Add the remaining 32 partial sums using a single warp
    if(threadIdx.x < WARP_SIZE) {
      temp = buffer[threadIdx.x];
      for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
        temp += __shfl_xor_sync(0xffffffff,temp, delta);
    }

    // Add this block's sum to the total sum
    if(threadIdx.x == 0)
        atomic_add(x+j, temp);
  }
}

void err_chk(cudaError err) {
  if (err != cudaSuccess) {
    cout << cudaGetErrorString(err) << endl;
    assert(false);
  }
}

int main() {

  
  double* x = new double[d]();
  double* z = new double[d*n];

  for (int i = 0; i < n * d; i ++)
    z[i] = i;

  double* d_x;
  double* d_z;

  err_chk(cudaMalloc(&d_x, sizeof(double)*d));
  err_chk(cudaMalloc(&d_z, sizeof(double)*n*d));
  err_chk(cudaMemcpy(d_z, z, sizeof(double)*n*d, cudaMemcpyHostToDevice));
                                        
  parallel_sum <<<n / 1024,1024>>> (d_z, d_x);

  err_chk(cudaMemcpy(x, d_x, sizeof(double)*d, cudaMemcpyDeviceToHost));
  
  cudaFree(d_z);
  cudaFree(d_x);
  
  for (int i = 0; i < 10; i++)
    printf("%.1f\n", x[i]);
  
  return 0;
}


    
