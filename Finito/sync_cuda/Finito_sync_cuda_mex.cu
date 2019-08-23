#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include "mex.h"
#include "matrix.h"
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
  } while (assumed != old);
  return __longlong_as_double(old);
}
#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      printf("Message: %s\n", cudaGetErrorString(x));   \
      assert(false);}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {    \
      printf("CuRand error at %s:%d\n",__FILE__,__LINE__);      \
      assert(false);}} while(0)

#define WARP_SIZE 32

using namespace std;
using namespace std::chrono;

__global__ void zCalculate(const double* __restrict__ x_a,
                           const double* __restrict__ y,
                           const double* __restrict__ z_a,
                           const double* __restrict__ mean_z,
                           double* __restrict__ delta_z,
                           const unsigned int* __restrict__ random_index,
                           int itr,
                           int n, int dim, double alpha, double s, int NUM_AGENT)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int lane = threadIdx.x % WARP_SIZE; // TODO: make sure that WARP_SIZE divides blockDim.x 
  const int warpIdx = idx / WARP_SIZE;
  const int ik =  random_index[itr * NUM_AGENT + warpIdx] % n;

  // Use of coalesced memory access in light of random index access
  
  // One option is to have a single warp access consecutive datapoints
  // (circularly consecutive, so use mod (%) to wrap around) by having
  // only the 0th thread in the warp generate a random index and
  // sharing it among the threads. This option is implemented

  // Another option is to have the 32 threads within a single warp
  // process the same datapoint
  
  // Correct. Do we need syncwarp?
  // The time spent is exactly the same as direct use of mean_z.
  // It seems that compiler did it for us
  // __shared__ double s_mean_z[dim];
  // for (int c = lane; c < dim; c+=WARP_SIZE)
  //   s_mean_z[c] = mean_z[c];
  // __syncwarp();

  // Correct. Slower than previous one. Pessimization
  // __shared__ double s_mean_z[dim];
  // for (int c = 0; c < dim; c++)
  //   s_mean_z[c] = mean_z[c];
  // __syncwarp();
  
  double dot = 0;
  for (int c = lane; c < dim; c+=WARP_SIZE) 
    dot += mean_z[c] * x_a[dim * ik + c];
  
  __syncwarp();
  // Sum up all "dot" across warps. Store the result in variable "dot" in *every* thread using xor
  for (int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
    dot += __shfl_xor_sync(0xffffffff, dot, delta);
  
  for (int c =  lane; c < dim; c+=WARP_SIZE) {        
    delta_z[warpIdx * dim + c] = mean_z[c] - z_a[ik * dim + c] - 
      alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_a[dim * ik + c] + s * mean_z[c]);
  }
}

    
__global__ void zUpdate(double* __restrict__ z_a,
                        double* __restrict__ delta_z,
                        unsigned int* __restrict__ random_index,
                        int itr,
                        int n, int dim, int NUM_AGENT)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int lane = idx % WARP_SIZE; // TODO: threadIdx % WARP_SIZE
  const int warpIdx = idx / WARP_SIZE;
  const int ik =  random_index[itr * NUM_AGENT + warpIdx] % n;
  //XXX Non-coalesced memory access XXX
  for (int c = lane; c < dim; c+=WARP_SIZE) {
    // z_a[ik * dim + c] +=  delta_z[warpIdx * dim + c];
    // atomic gives 50ms performance loss in total for a medium dataset
    
    // int cc = (c + warpIdx * 32) % dim;
    // TODO: This gives only 5ms performance gain; I am not sure why it's so small
    
    atomicAdd(&z_a[ik * dim + c], delta_z[warpIdx * dim + c]); 
  }
}


__global__ void mean_zUpdate (const double* __restrict__ delta_mean_z,
                              double* __restrict__ mean_z,
                              int len) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < len)
    mean_z[idx] += delta_mean_z[idx];
}


__global__ void reduction_sum_divided(const double* __restrict__ v,
                                     double* __restrict__ sum_v,
                                     int num_row, int num_col, double div) {
  // Lauch num_col threads in total
  
  // Holds intermediates in shared memory reduction
  __syncthreads();
  __shared__ double buffer[1024/WARP_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % WARP_SIZE;

  for (int k = 0; k < num_row; k++) {
    int j = (k + blockIdx.x) % num_row;
    //j = k;
    double temp;
    // All threads in a block of 1024 take an element
    temp = v[i + num_col * j];
    
    // All warps in this block (32) compute the sum of all
    // threads in their warp
    for(int delta = WARP_SIZE/2; delta > 0; delta /= 2)
      temp += __shfl_xor_sync(0xffffffff, temp, delta);

    // Write all 32 of these partial sums to shared memory
    if(lane == 0)
      buffer[threadIdx.x / WARP_SIZE] = temp / div;
    
    __syncthreads();

    // Add the remaining 32 partial sums using a single warp
    if(threadIdx.x < WARP_SIZE) {
      temp = buffer[threadIdx.x];
      for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
        temp += __shfl_xor_sync(0xffffffff,temp, delta);
    }

    // Add this block's sum to the total sum
    if(threadIdx.x == 0)
      atomicAdd(sum_v+j, temp);  
    // sum_v[j] += temp;
  }
}


__global__ void parallel_sum_divided(const double* __restrict__ v,
                                     double* __restrict__ sum_v,
                                     int num_row, int num_col, double div) {
  // Lauch num_col threads in total
  const int idx = blockIdx.x * blockDim.x + threadIdx.x; // 1 ~ num_col
  if (idx < num_col) {
    double total = 0;
    for (int c = 0; c < num_row; c++) {
      total += v[idx + c * num_col];
    }
    sum_v[idx] = total / div;
  }
}


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  const int n = mxGetDimensions(prhs[1])[0];
  const int dim = mxGetDimensions(prhs[0])[0] /  n;
  
  const double* x_a = mxGetPr(prhs[0]);
  const double* y = mxGetPr(prhs[1]);
  const double alpha = *mxGetPr(prhs[2]);
  const double s = *mxGetPr(prhs[3]);
  const int epoch = *mxGetPr(prhs[4]);
  const int NUM_AGENT = *mxGetPr(prhs[5]);
  const int BLOCKSIZE = *mxGetPr(prhs[6]);

  double* z_a = mxGetPr(prhs[7]);
  double* mean_z = mxGetPr(prhs[8]);
  
  double *d_x_a, *d_y;
  CUDA_CALL(cudaMalloc(&d_x_a, sizeof(double) * n * dim));
  CUDA_CALL(cudaMalloc(&d_y, sizeof(double) * n ));
  CUDA_CALL(cudaMemcpy(d_x_a, x_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
  
  double *d_z_a, *d_mean_z;
  CUDA_CALL(cudaMalloc(&d_z_a, sizeof(double) * n * dim));
  CUDA_CALL(cudaMalloc(&d_mean_z, sizeof(double) * dim));                 
  CUDA_CALL(cudaMemcpy(d_z_a, z_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));

  double* d_delta_z;
  CUDA_CALL(cudaMalloc(&d_delta_z, sizeof(double) * dim * NUM_AGENT));

  double* delta_mean_z = new double [dim];
  double* d_delta_mean_z;
  CUDA_CALL(cudaMalloc(&d_delta_mean_z, sizeof(double) * dim));

  unsigned int * d_random_index;
  auto now_clock = time_point_cast<milliseconds>(system_clock::now());
  auto seed = now_clock.time_since_epoch().count();
  CUDA_CALL(cudaMalloc(&d_random_index, sizeof(unsigned int) * n * epoch));
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
  CURAND_CALL(curandGenerate(gen, d_random_index, n * epoch));
  
  duration <double> elapsed;
  cudaDeviceSynchronize();
  auto start = high_resolution_clock::now();
  
  for (int k = 0; k < epoch * n / NUM_AGENT; k++) {
    memset(delta_mean_z, 0, sizeof(double) * dim);
    CUDA_CALL(cudaMemcpy(d_delta_mean_z, delta_mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));
    
    zCalculate <<< NUM_AGENT * WARP_SIZE / BLOCKSIZE, BLOCKSIZE>>>
      (d_x_a, d_y, d_z_a, d_mean_z, d_delta_z, d_random_index, k,
       n, dim, alpha, s, NUM_AGENT);
    
    zUpdate <<< NUM_AGENT * WARP_SIZE / BLOCKSIZE, BLOCKSIZE>>>
      (d_z_a, d_delta_z, d_random_index, k,
       n, dim, NUM_AGENT);
    //--------The following code enforce z_mean consistency somehow inefficiently-----------
    // memset(mean_z, 0, sizeof(double) * dim);
    // CUDA_CALL(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));  // < 0.01s
    // reduction_sum_divided <<< n / 1024, 1024>>> (d_z_a, d_mean_z, dim, n, n); // 0.35 s 
    //---------------------------------------------------------------

    //------------------------One way to calculate delta_mean_z-------------------------

    // reduction_sum_divided <<< NUM_AGENT / 1024, 1024>>>
    //   (d_delta_z, d_delta_mean_z, dim, NUM_AGENT, n); // 0.35 s

    //------------------Another way to calculate delta_mean_z----------------------------
    parallel_sum_divided <<< 1 + (dim - 1) / BLOCKSIZE, BLOCKSIZE>>>
      (d_delta_z, d_delta_mean_z, NUM_AGENT, dim, n);
                                  
    //---------------------------------------------------------------------------------

    //---------------Comment out the following code when enforcing z_mean consistency------------

    mean_zUpdate <<< 1 + (dim - 1) / BLOCKSIZE, BLOCKSIZE>>>
      (d_delta_mean_z, d_mean_z, dim);
    //-------------------------------------------------------------------------------------------
    
  }
  cudaDeviceSynchronize();
  auto end = high_resolution_clock::now();
  elapsed = end - start;
  
  CUDA_CALL(cudaMemcpy(mean_z, d_mean_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(z_a, d_z_a, sizeof(double) * dim * n, cudaMemcpyDeviceToHost));

  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(1, n * dim, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  double * ptr0 = mxGetPr(plhs[0]);
  double * ptr1 = mxGetPr(plhs[1]);
  double * ptr2 = mxGetPr(plhs[2]);
  
  for (int c = 0; c < dim; c++)
    ptr0[c] = mean_z[c];

  for (int c = 0; c < dim * n; c++)
    ptr1[c] = z_a[c];

  *ptr2 = elapsed.count();
  
  cudaFree(d_z_a);
  cudaFree(d_mean_z);
  cudaFree(d_x_a);
  cudaFree(d_y);
  cudaFree(d_delta_mean_z);
  cudaFree(d_delta_z);

  delete [] delta_mean_z; 
}