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

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
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

__global__ void run_async(const double* __restrict__ x_a,
               const double* __restrict__ y,
               double* z,
               double* dots,
               curandState* states,
               int* itr_ptr,
               int n, int dim, double alpha, double s, int max_itr
               )
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int lane = threadIdx.x % WARP_SIZE; // TODO: make sure that WARP_SIZE divides blockDim.x 
  const int warpIdx = idx / WARP_SIZE;
  if (lane == 0)
    curand_init(0, warpIdx, 0, &states[warpIdx]);
  
  while (*itr_ptr < max_itr) {
    int ik;
    if (lane == 0) {
      atomicAdd(itr_ptr, 1);
      ik = curand(&states[warpIdx]) % dim;
    }
    ik = __shfl_sync(0xffffffff, ik, 0);

    double delta_z = 0;
    for (int r = lane; r < n; r += WARP_SIZE) {
      delta_z += -1.0 / (1+exp(y[r] * dots[r])) * y[r] * x_a[r + ik * n];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
      delta_z += __shfl_xor_sync(0xffffffff, delta_z, offset);
    __syncwarp();
    
    delta_z = - alpha * ( delta_z / n + s * z[ik]);

    __syncwarp();

    
    for (int r = lane; r < n; r+=WARP_SIZE)
      atomic_add(&dots[r], delta_z * x_a[r + ik * n]);
    
    if (lane == 0){
      atomic_add(&z[ik], delta_z);
    }

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
  const int max_itr = *mxGetPr(prhs[4]);
  const int NUM_AGENT = *mxGetPr(prhs[5]);
  const int BLOCKSIZE = *mxGetPr(prhs[6]);
  double* z = mxGetPr(prhs[7]);

  double *d_x_a, *d_y;
  CUDA_CALL(cudaMalloc(&d_x_a, sizeof(double) * n * dim));
  CUDA_CALL(cudaMalloc(&d_y, sizeof(double) * n ));
  CUDA_CALL(cudaMemcpy(d_x_a, x_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));

  double *d_z;
  CUDA_CALL(cudaMalloc(&d_z, sizeof(double) * dim));             
  CUDA_CALL(cudaMemcpy(d_z, z, sizeof(double) * dim, cudaMemcpyHostToDevice));

  int zero = 0;
  int* d_itr_ptr;
  CUDA_CALL(cudaMalloc(&d_itr_ptr, sizeof(int)));
  CUDA_CALL(cudaMemcpy(d_itr_ptr, &zero, sizeof(int), cudaMemcpyHostToDevice));

  curandState *d_states;
  CUDA_CALL(cudaMalloc(&d_states, sizeof(curandState) * NUM_AGENT));

  double *d_dots, *dots;
  CUDA_CALL(cudaMalloc(&d_dots, sizeof(double) * n));
  dots = new double [n]();
  CUDA_CALL(cudaMemcpy(d_dots, dots, sizeof(double) * n, cudaMemcpyHostToDevice));
  
  chrono :: duration <double> elapsed (0);
  chrono :: high_resolution_clock :: time_point start, end;

  cudaDeviceSynchronize();start=chrono::high_resolution_clock::now();

  run_async <<< NUM_AGENT * WARP_SIZE / BLOCKSIZE, BLOCKSIZE >>>
    (d_x_a, d_y, d_z, d_dots, d_states, d_itr_ptr,
     n, dim, alpha, s, max_itr);
 
  cudaDeviceSynchronize();end=chrono::high_resolution_clock::now();elapsed=end-start;

  // cout<< endl
  //     <<"NUM_AGENT: " << NUM_AGENT << endl
  //     <<"BLOCKSIZE: " << BLOCKSIZE << endl
  //     <<"elapsed time: "<<elapsed.count()<<" s"<<endl
  //     << endl;

  CUDA_CALL(cudaMemcpy(z, d_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));

  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);

  double * ptr0 = mxGetPr(plhs[0]);

  for (int c = 0; c < dim; c++)
    ptr0[c] = z[c];
  
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *ptr1 = mxGetPr(plhs[1]);
  *ptr1 = elapsed.count();
  
  cudaFree(d_x_a);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_itr_ptr);
  cudaFree(d_dots);
  cudaFree(d_states);
  delete[] dots;
}