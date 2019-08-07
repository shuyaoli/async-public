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

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      printf("Message: %s\n", cudaGetErrorString(x));   \
      assert(false);}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {    \
      printf("CuRand error at %s:%d\n",__FILE__,__LINE__);      \
      assert(false);}} while(0)

#define WARP_SIZE 32

using namespace std;

__global__ void dotsCalculate (const double* __restrict__ x_a,
                               const double* __restrict__ delta_z,
                               const unsigned int* __restrict__ random_index,
                               double* dots,
                               int itr,
                               int n, int dim, int NUM_AGENT) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    for (int i = 0; i < NUM_AGENT; i++) {
      int ik = random_index[itr * NUM_AGENT + i] % dim;
      dots[idx] += delta_z[i] * x_a[idx + ik * n];
    }
  }
}


__global__ void zCalculate(const double* __restrict__ x_a,
                           const double* __restrict__ y,
                           const double* __restrict__ z,
                           double* delta_z,
                           const unsigned int* __restrict__ random_index,
                           const double* __restrict__ dots,
                           int itr,
                           int n, int dim, double alpha, double s, int NUM_AGENT)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int lane = threadIdx.x % WARP_SIZE; // TODO: make sure that WARP_SIZE divides blockDim.x 
  const int warpIdx = idx / WARP_SIZE;
  const int ik =  random_index[itr * NUM_AGENT + warpIdx] % dim;
  const double z_ik = z[ik];
  
  double result = 0;

  for (int r = lane; r < n; r += WARP_SIZE) {
    result += -1.0 / (1+exp(y[r] * dots[r])) * y[r] * x_a[r + ik * n];
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    result += __shfl_down_sync(0xffffffff, result, offset);

  result = - alpha * (result / n + s * z_ik);
  if (lane == 0)
    delta_z[warpIdx] = result;
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

  double* delta_z = new double[NUM_AGENT];
  double* d_delta_z;
  CUDA_CALL(cudaMalloc(&d_delta_z, sizeof(double) * NUM_AGENT));

  chrono :: duration <double> elapsed (0);

  unsigned int *random_index, *d_random_index;
  random_index = new unsigned int[max_itr];
  CUDA_CALL(cudaMalloc(&d_random_index, sizeof(unsigned int) * max_itr));
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL)); // TODO: seed with time
  CURAND_CALL(curandGenerate(gen, d_random_index, max_itr));
  CUDA_CALL(cudaMemcpy(random_index, d_random_index, sizeof(unsigned int) * max_itr, cudaMemcpyDeviceToHost));

  double *d_dots, *dots;
  CUDA_CALL(cudaMalloc(&d_dots, sizeof(double) * n));
  dots = new double [n]();
  CUDA_CALL(cudaMemcpy(d_dots, dots, sizeof(double) * n, cudaMemcpyHostToDevice));

  
  cudaDeviceSynchronize(); auto start = chrono :: high_resolution_clock::now();


  for (int k = 0; k < max_itr / NUM_AGENT; k++) {
    zCalculate <<< NUM_AGENT * WARP_SIZE / BLOCKSIZE, BLOCKSIZE>>>
      (d_x_a, d_y, d_z, d_delta_z, d_random_index, d_dots, k,
       n, dim, alpha, s, NUM_AGENT);
  
    dotsCalculate <<< 1 + (n - 1) / BLOCKSIZE, BLOCKSIZE >>>
      (d_x_a, d_delta_z, d_random_index, d_dots, k, n, dim, NUM_AGENT);
    
    CUDA_CALL(cudaMemcpy(delta_z, d_delta_z, sizeof(double) * NUM_AGENT, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(z, d_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_AGENT; i++) {
      int ik = random_index[k * NUM_AGENT + i] % dim;
      z[ik] += delta_z[i];
    }
    
    CUDA_CALL(cudaMemcpy(d_z, z, sizeof(double) * dim, cudaMemcpyHostToDevice));    
  }
  cudaDeviceSynchronize(); auto end = chrono::high_resolution_clock::now(); elapsed += end - start;

  cout<< endl
      <<"NUM_AGENT: " << NUM_AGENT << endl
      <<"BLOCKSIZE: " << BLOCKSIZE << endl
      <<"elapsed time: "<<elapsed.count()<<" s"<<endl
      << endl;
  
  CUDA_CALL(cudaMemcpy(z, d_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));

  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);

  double * ptr0 = mxGetPr(plhs[0]);

  for (int c = 0; c < dim; c++)
    ptr0[c] = z[c];
  
  cudaFree(d_x_a);
  cudaFree(d_y);
  cudaFree(d_delta_z);
  cudaFree(d_random_index);
  cudaFree(d_dots);

  delete[] delta_z;
  delete[] random_index;
}