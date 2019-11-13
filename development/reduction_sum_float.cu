#include <iostream>
#include <cassert>
#include <ctime>
#include <stdio.h>
#include <cuda.h>

#define n 131072 
#define d 512
#define WARP_SIZE 32


using namespace std;
void err_chk(cudaError err) {
  if (err != cudaSuccess) {
    cout << cudaGetErrorString(err) << endl;
    assert(false);
  }
}

__global__ void parallel_sum(const float* __restrict__ z, float *x_half) {
  //Holds intermediates in shared memory reduction
  __syncthreads();

  __shared__ float buffer[1024/WARP_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % WARP_SIZE;

  for (int k=0; k<d; k++) {
    int j = (k + blockIdx.x) % d;
    //j = k;
    float temp;
    // All threads in a block of 1024 take an element
    temp = z[i + n*j];
    
    // All warps in this block (32) compute the sum of all
    // threads in their warp
    for(int delta = WARP_SIZE/2; delta > 0; delta /= 2)
      temp += __shfl_xor(temp, delta);


    // Write all 32 of these partial sums to shared memory
    if(lane == 0)
      buffer[threadIdx.x / WARP_SIZE] = temp;
    
    __syncthreads();


    // Add the remaining 32 partial sums using a single warp
    if(threadIdx.x < WARP_SIZE) {
      temp = buffer[threadIdx.x];
      for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
        temp += __shfl_xor(temp, delta);
    }

    // Add this block's sum to the total sum
    if(threadIdx.x == 0)
      atomicAdd(x_half+j, temp);
  }
}

int main() {

  
  float* x_half = new float[d]();
  float* z = new float[d*n];

  // memset(x_half,0,d*sizeof(float));
  for (int i = 0; i < n * d; i ++)
    z[i] = i;

  float* d_x_half;
  float* d_z;

  err_chk(cudaMalloc(&d_x_half, sizeof(float)*d));
  err_chk(cudaMalloc(&d_z, sizeof(float)*n*d));

  
  err_chk(cudaMemcpy(d_z, z, sizeof(float)*n*d, cudaMemcpyHostToDevice));
  err_chk(cudaMemcpy(d_x_half, x_half, sizeof(float)*d, cudaMemcpyHostToDevice));
                                        
  parallel_sum <<<n/1024,1024>>> (d_z, d_x_half);

  err_chk(cudaMemcpy(x_half, d_x_half, sizeof(float)*d, cudaMemcpyDeviceToHost));


  for (int i = 0; i < n * d; i ++)
    z[i] = 2 * i;

  err_chk(cudaMemcpy(d_z, z, sizeof(float)*n*d, cudaMemcpyHostToDevice));
  memset(x_half, 0, sizeof(float) * d);
  // for (int i = 0; i < d; i++)
  //   x_half[i] = 0;
  err_chk(cudaMemcpy(d_x_half, x_half, sizeof(float)*d, cudaMemcpyHostToDevice));
  parallel_sum <<<n/1024,1024>>> (d_z, d_x_half);
  err_chk(cudaMemcpy(x_half, d_x_half, sizeof(float)*d, cudaMemcpyDeviceToHost));

  
  cudaFree(d_z);
  cudaFree(d_x_half);
  for (int i = 0; i < 10; i++)
    printf("%.0f\n", x_half[i]);
  return 0;
}