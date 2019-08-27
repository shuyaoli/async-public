#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include "mex.h"
#include "matrix.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      printf("Message: %s\n", cudaGetErrorString(x));   \
      assert(false);}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {    \
      printf("CuRand error at %s:%d\n",__FILE__,__LINE__);      \
      assert(false);}} while(0)

#define WARP_SIZE 32

class myMutex {
private:
  int _m;
public:
  myMutex () {_m = 0;}
  __device__ void lock (){
    while ( atomicExch(&_m, 1) == 1) {}
  };
  __device__ void unlock() {_m = 0;};
};

using namespace std;
using namespace chrono;

__global__ void run_async(const double* __restrict__ x_a,
                          const double* __restrict__ y,
                          double* z_a,
                          double* mean_z,
                          curandState* states,
                          int* itr_ptr,
                          int n, int dim, double alpha, double s, int epoch,
                          long long seed)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int lane = threadIdx.x % WARP_SIZE; // TODO: threadIdx % WARP_SIZE
  const int warpIdx = idx / WARP_SIZE;
  double delta_buffer;
  
  if (lane == 0) {
    curand_init(seed, warpIdx, 0, &states[warpIdx]);
  }
  
  while (*itr_ptr < epoch * n) {
    int ik;
    if (lane == 0) {
      atomicAdd(itr_ptr, 1);
      ik = curand(&states[warpIdx]) % n;
    }
    ik = __shfl_sync(0xffffffff, ik, 0);

    double dot = 0;
    for (int c = lane; c < dim; c+=WARP_SIZE) 
      dot += mean_z[c] * x_a[dim * ik + c];
  
    __syncwarp();
    for (int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
      dot += __shfl_xor_sync(0xffffffff, dot, delta);

    for (int c =  lane; c < dim; c+=WARP_SIZE) {
      // Intention: different warps can start from different coordinates to avoid collision
      // int d = (c + warpIdx * WARP_SIZE) % dim;
      // Doesn't work, basically no speed up.

      // ------------------------------------------------------------------------------------------
      // Those could be done with three for loops, but then we need to remember the delta_buffer for `dim` different coordinates.
      // Each block has multiple agents (1-32). A typical mimimum is 4 agents, in which case one block has 128 threads.
      // Shared memory is not large enough to store 4 * `dim` * 64 bit when dim is larger than 1500.
      // It follows that if we use three loops, delta_buffer has to reside in global memory, which is very slow.
      delta_buffer = mean_z[c] - z_a[ik * dim + c] - 
        alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_a[dim * ik + c] + s * mean_z[c]);
      atomicAdd(&z_a[ik * dim + c], delta_buffer);
      atomicAdd(&mean_z[c], delta_buffer / n);
    }

    
  }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // prhs[0] and prhs[1] stores x_a and y respectively
  const int n = mxGetDimensions(prhs[1])[0];
  const int dim = mxGetDimensions(prhs[0])[0] /  n;
  
  const double* x_a = mxGetPr(prhs[0]);
  const double* y = mxGetPr(prhs[1]);
  const double alpha = *mxGetPr(prhs[2]);
  const double s = *mxGetPr(prhs[3]);
  const int epoch = *mxGetPr(prhs[4]);
  const int NUM_AGENT = *mxGetPr(prhs[5]);
  const int BLOCKSIZE = *mxGetPr(prhs[6]);
  
  double *d_x_a, *d_y;
  CUDA_CALL(cudaMalloc(&d_x_a, sizeof(double) * n * dim));
  CUDA_CALL(cudaMalloc(&d_y, sizeof(double) * n ));
  CUDA_CALL(cudaMemcpy(d_x_a, x_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
  
  double* z_a =  new double[n * dim]();
  double* mean_z = new double [dim]();
  double *d_z_a, *d_mean_z;
  CUDA_CALL(cudaMalloc(&d_z_a, sizeof(double) * n * dim));
  CUDA_CALL(cudaMalloc(&d_mean_z, sizeof(double) * dim));                 
  CUDA_CALL(cudaMemcpy(d_z_a, z_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));

  int zero = 0;
  int* d_itr_ptr;
  CUDA_CALL(cudaMalloc(&d_itr_ptr, sizeof(int)));
  CUDA_CALL(cudaMemcpy(d_itr_ptr, &zero, sizeof(int), cudaMemcpyHostToDevice));

  curandState *d_states;
  CUDA_CALL(cudaMalloc(&d_states, sizeof(curandState) * NUM_AGENT));
  auto now_clock = time_point_cast<milliseconds>(system_clock::now());
  auto seed = now_clock.time_since_epoch().count();
  
  duration <double> elapsed (0);
  high_resolution_clock :: time_point start, end;

  cudaDeviceSynchronize();
  start = high_resolution_clock::now();
  
  run_async <<< NUM_AGENT * WARP_SIZE / BLOCKSIZE, BLOCKSIZE>>>
    (d_x_a, d_y, d_z_a, d_mean_z,  d_states, d_itr_ptr,
     n, dim, alpha, s, epoch, seed);
  
  cudaDeviceSynchronize();
  end = chrono::high_resolution_clock::now();
  elapsed = end - start;
  
  CUDA_CALL(cudaMemcpy(mean_z, d_mean_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));
  
  // MATLAB Output
  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);
  double * ptr0 = mxGetPr(plhs[0]);
  
  for (int c = 0; c < dim; c++)
    ptr0[c] = mean_z[c];
  
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *ptr1 = mxGetPr(plhs[1]);
  *ptr1 = elapsed.count();
  cudaFree(d_itr_ptr);
  cudaFree(d_states);
  cudaFree(d_z_a);
  cudaFree(d_mean_z);
  cudaFree(d_x_a);
  cudaFree(d_y);
  delete []mean_z;
  delete []z_a;
}