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
using namespace chrono;
// void err_chk(cudaError err) {
//   if (err != cudaSuccess) {
//     cout << cudaGetErrorString(err) << endl;
//     assert(false);
//   }
// }
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
      // int d = (c + warpIdx * WARP_SIZE) % dim;
      // TODO: it doesn't work...basically no speed up
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
  // const char * const errId = "Finito_async_cuda_mex:InvalidInput";
  // const char * const errMsg = "Invalid input to MEX file.";

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
  
  // cout<< endl
  //     <<"NUM_AGENT: " << NUM_AGENT << endl
  //     <<"BLOCKSIZE: " << BLOCKSIZE << endl
  //     <<"elapsed time: "<<elapsed.count()<<" s"<<endl
  //     << endl;
  
  // MATLAB Output
  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);
  double * ptr0 = mxGetPr(plhs[0]);
  
  for (int c = 0; c < dim; c++)
    ptr0[c] = mean_z[c];
  
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *ptr1 = mxGetPr(plhs[1]);
  *ptr1 = elapsed.count();
  
  cudaFree(d_states);
  cudaFree(d_z_a);
  cudaFree(d_mean_z);
  cudaFree(d_x_a);
  cudaFree(d_y);
  delete []mean_z;
  delete []z_a;
}