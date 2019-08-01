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
#include "gpu/mxGPUArray.h"
#include "matrix.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      printf("Message: %s\n", cudaGetErrorString(x));   \
      return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {    \
      printf("CuRand error at %s:%d\n",__FILE__,__LINE__);      \
      return EXIT_FAILURE;}} while(0)

#define WARP_SIZE 32

using namespace std;
void err_chk(cudaError err) {
  if (err != cudaSuccess) {
    cout << cudaGetErrorString(err) << endl;
    assert(false);
  }
}
__global__ void run_async(const double* __restrict__ x_a,
                          const double* __restrict__ y,
                          double* z_a,
                          double* mean_z,
                          double* delta_z,
                          curandState* states,
                          int* itr_ptr,
                          int n, int dim, int alpha, int s, int epoch)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int lane = threadIdx.x % WARP_SIZE; // TODO: threadIdx % WARP_SIZE
  const int warpIdx = idx / WARP_SIZE;
  double delta_buffer;
  
  if (lane == 0)
    curand_init(0, warpIdx, 0, &states[warpIdx]);
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
  const char * const errId = "Finito_async_cuda_mex:InvalidInput";
  const char * const errMsg = "Invalid input to MEX file.";

  // prhs[0] and prhs[1] stores x_a and y respectively
  const double alpha = *mxGetPr(prhs[2]);
  const double s = *mxGetPr(prhs[3]);
  const int epoch = *mxGetPr(prhs[4]);
  const int NUM_AGENT = *mxGetPr(prhs[5]);
  const int BLOCKSIZE = *mxGetPr(prhs[6]);
  
  const mxGPUArray *x_a, *y;
  const double *d_x_a, *d_y;

  mxInitGPU();

  if (!(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1]))) {
    mexErrMsgIdAndTxt(errId,errMsg);
  }

  x_a = mxGPUCreateFromMxArray(prhs[0]);
  y = mxGPUCreateFromMxArray(prhs[1]);
  
  if (mxGPUGetClassID(x_a) != mxDOUBLE_CLASS ||
      mxGPUGetClassID(y) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
  }

  d_x_a = (const double*)(mxGPUGetDataReadOnly(x_a));
  d_y = (const double*)(mxGPUGetDataReadOnly(y));

  const int n = (int)(mxGPUGetNumberOfElements(y));
  const int dim = (int)(mxGPUGetNumberOfElements(x_a)) / n;
  
  double* z_a =  new double[n * dim]();
  double* mean_z = new double [dim]();
  double *d_z_a, *d_mean_z;
  err_chk(cudaMalloc(&d_z_a, sizeof(double) * n * dim));
  err_chk(cudaMalloc(&d_mean_z, sizeof(double) * dim));                 
  err_chk(cudaMemcpy(d_z_a, z_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  err_chk(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));

  int zero = 0;
  int* d_itr_ptr;
  err_chk(cudaMalloc(&d_itr_ptr, sizeof(int)));
  err_chk(cudaMemcpy(d_itr_ptr, &zero, sizeof(int), cudaMemcpyHostToDevice));

  double* d_delta_z;
  curandState *d_states;
  
  err_chk(cudaMalloc(&d_delta_z, sizeof(double) * dim * NUM_AGENT));
  err_chk(cudaMalloc(&d_states, sizeof(curandState) * NUM_AGENT));
  
  chrono :: duration <double> elapsed (0);
  chrono :: high_resolution_clock :: time_point start, end;
  cudaDeviceSynchronize();start=chrono::high_resolution_clock::now();
  run_async <<< NUM_AGENT * WARP_SIZE / BLOCKSIZE, BLOCKSIZE>>>
    (d_x_a, d_y, d_z_a, d_mean_z, d_delta_z, d_states, d_itr_ptr,
     n, dim, alpha, s,epoch);
  cudaDeviceSynchronize();end=chrono::high_resolution_clock::now();elapsed=end-start;
  
  err_chk(cudaMemcpy(mean_z, d_mean_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));

  for (int i = 0; i < 4; i++) printf("%.15f\n", mean_z[i]);
  
  cout<<"NUM_AGENT: " << NUM_AGENT << endl
      <<"BLOCKSIZE: " << BLOCKSIZE << endl
      <<"elapsed time: "<<elapsed.count()<<" s"<<endl
      << endl;
  
  // MATLAB Output
  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);
  double * ptr = mxGetPr(plhs[0]);
  
  for (int c = 0; c < dim; c++)
    ptr[c] = mean_z[c];

  delete []mean_z;
  delete []z_a;

  mxGPUDestroyGPUArray(x_a);
  mxGPUDestroyGPUArray(y);
}