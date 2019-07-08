#include <iostream>
#include <cassert>
#include <ctime>
#include <stdio.h>
#include <cuda.h>


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

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

using namespace std;

#define n 131072 
#define d 512
#define WARP_SIZE 32
#define num_T double
#define IS_DOUBLE 1

// typedef float num_T;

__global__ void parallel_sum(const num_T* __restrict__ z, num_T *x_half) {
  //Holds intermediates in shared memory reduction
  __syncthreads();
  __shared__ num_T buffer[1024/WARP_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % WARP_SIZE;

  for (int k=0; k<d; k++) {
    int j = (k + blockIdx.x) % d;
    //j = k;
    num_T temp;
    // All threads in a block of 1024 take an element
    temp = z[i + n*j];
    
    // All warps in this block (32) compute the sum of all
    // threads in their warp
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
# if IS_DOUBLE
        atomic_add(x_half+j, temp);
# else
        atomicAdd(x_half+j, temp);
#endif
      // x_half[j] = temp;
  }
}

void err_chk(cudaError err) {
  if (err != cudaSuccess) {
    cout << cudaGetErrorString(err) << endl;
    assert(false);
  }
}

int main() {

  
  num_T* x_half = new num_T[d]();
  num_T* z = new num_T[d*n];

  // memset(x_half,0,d*sizeof(num_T));
  for (int i = 0; i < n * d; i ++)
    z[i] = i;

  num_T* d_x_half;
  num_T* d_z;

  err_chk(cudaMalloc((void**)&d_x_half, sizeof(num_T)*d));
  err_chk(cudaMalloc((void**)&d_z, sizeof(num_T)*n*d));
  err_chk(cudaMemcpy(d_z, z, sizeof(num_T)*n*d, cudaMemcpyHostToDevice));
    

  err_chk(cudaMemcpy(d_x_half, x_half, sizeof(num_T)*d, cudaMemcpyHostToDevice));
                                        
  parallel_sum <<<n/1024,1024>>> (d_z, d_x_half);

  err_chk(cudaMemcpy(x_half, d_x_half, sizeof(num_T)*d, cudaMemcpyDeviceToHost));

  for (int i = 0; i < 10; i++)
    printf("%f\n", x_half[i]);
  return 0;
}


    
