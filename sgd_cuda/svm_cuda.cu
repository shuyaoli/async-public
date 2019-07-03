#include <iostream>
#include <cassert>
#include <ctime>
#include <stdio.h>
#include "mat.h"
#define n 131072
#define d 512
#define WARP_SIZE 32
#define alpha 0.1f
#define lambda 0.1f
using std::cout;
using std::endl;
__global__ void zUpdate(const float* __restrict__ A, const float* __restrict__ b, const float* __restrict__ x_half, const float* __restrict__ inv_a_norm2, float* z);
__global__ void parallel_sum(const float* __restrict__ z, float *x_half);
void err_chk(cudaError err) {
  if (err != cudaSuccess) {
    cout << cudaGetErrorString(err) << endl;
    assert(false);
  }
}
int main() {
  MATFile *pmat;
  pmat = matOpen("svm_data.mat", "r");
  if (pmat == NULL) {
    printf("Error reopening file %s\n", file);
    assert(false);
  }
  mxArray* p_A;
  p_A = matGetVariable(pmat, "A");
  if (p_A == NULL) {
    printf("Error reading A\n");
    assert(false)
  }
  mxArray* p_y;
  p_y = matGetVariable(pmat, "y");
  if (p_y == NULL) {
    printf("Error reading y\n");
    assert(false)
  }
  const int max_iter = 1000;
  
  float* A = new float[n*d];
  double* double_A = mxGetPr(p_A);
  for (int ii=0; ii<n*d; ii++) {
    A[ii] = float(double_A[ii]);
  }
  float* y = new float[n];
  double* double_y = mxGetPr(p_y);
  for (int ii=0; ii<n*d; ii++) {
    y[ii] = float(double_y[ii]);
  }
  mxDestroyArray(p_A);
  mxDestroyArray(p_y);
  if (matClose(pmat) != 0) {
    printf("Error closing file %s\n",file);
    return(EXIT_FAILURE);
  }
  float* x_half = new float[d];
  float* z = new float[d*n];
  memset(z,0,d*n*sizeof(float));
  float* inv_a_norm2 = new float[n];
  for (int i=0; i<n; i++) {
    inv_a_norm2[i] = 0.0;
    for (int j=0; j<d; j++) {
      inv_a_norm2[i] += A[i + n*j]*A[i + n*j];
    }
    inv_a_norm2[i] = 1/inv_a_norm2[i];
  }
  const float inv_n1al = 1/(float(n) * (1+alpha*lambda) );
  float* d_A;
  float* d_y;
  float* d_x_half;
  float* d_z;
  float* d_inv_a_norm2;
  err_chk(cudaMalloc((void**)&d_A, sizeof(float)*n*d));
  err_chk(cudaMalloc((void**)&d_y, sizeof(float)*n));
  err_chk(cudaMalloc((void**)&d_x_half, sizeof(float)*d));
  err_chk(cudaMalloc((void**)&d_z, sizeof(float)*n*d));
  err_chk(cudaMalloc((void**)&d_inv_a_norm2, sizeof(float)*n));
  cout << "sdjifosjd" << endl;  
  err_chk(cudaMemcpy(d_A, A, sizeof(float)*n*d, cudaMemcpyHostToDevice));
  err_chk(cudaMemcpy(d_y, y, sizeof(float)*n, cudaMemcpyHostToDevice));
  cout << "here" << endl;
  err_chk(cudaMemcpy(d_z, z, sizeof(float)*n*d, cudaMemcpyHostToDevice));
  err_chk(cudaMemcpy(d_inv_a_norm2, inv_a_norm2, sizeof(float)*n, cudaMemcpyHostToDevice));
  
cout << "starting iteration" << endl;
clock_t begin = clock();
for (int ii=0; ii<max_iter; ii++) {
    if (ii%100==0)
        cout << "At iteration count: " << ii << endl;
        
    memset(x_half,0,d*sizeof(float));
    err_chk(cudaMemcpy(d_x_half, x_half, sizeof(float)*d, cudaMemcpyHostToDevice));
    parallel_sum <<<n/1024,1024>>> (d_z,d_x_half);
    err_chk(cudaMemcpy(x_half, d_x_half, sizeof(float)*d, cudaMemcpyDeviceToHost));
    for (int j=0; j<d; j++) 
      x_half[j] *= inv_n1al;
    err_chk(cudaMemcpy(d_x_half, x_half, sizeof(float)*d, cudaMemcpyHostToDevice));
    
    
    //    z_half = 2*x_half*ones(1,n) - z;
    // beta = max(0,min(alpha,y.*((b+y)-diag(A*z_half))./norm_a));
    //x = (z_half+A'*diag(beta.*y));
    zUpdate <<< n/512, 512>>> (d_A, d_y, d_x_half, d_inv_a_norm2, d_z);
        
}
 cudaDeviceSynchronize();
   float runtime = float(clock() - begin) / CLOCKS_PER_SEC;
  cout << "Total runtime is " << runtime << "s." << endl;
    cout << "This is " << runtime / float(max_iter) * 1000 << "ms per iteration" << endl;
err_chk(cudaMemcpy(x_half, d_x_half, sizeof(float)*d, cudaMemcpyDeviceToHost));
MATFile *w_mat;
w_mat = matOpen(file, "w");
if (pmat == NULL) {
  printf("Error creating file %s\n", file);
  printf("(Do you have write permission in this directory?)\n");
  assert(false);
}
mxArray* w_x mxCreateDoubleMatrix(d,1,mxREAL);
double* double_w_x = mxGetPr(w_x);
for (int j=0; j<d; j++) {
  double_w_x[j] = double(x_half[j]);
}
status = matPutVariable(w_mat, "x_half", w_x);
if (status != 0) {
cout << "write fail" << endl;
  assert(false);
}
mxDestroyArray(w_x);
if (matClose(w_mat) != 0) {
printf("Error closing file\n");
return(EXIT_FAILURE);
}
  cudaFree(d_A);
  cudaFree(d_y);
  cudaFree(d_x_half);
  cudaFree(d_z);
  cudaFree(d_inv_a_norm2);
    
  delete[] A;
  delete[] y;
  delete[] x_half;     
  delete[] z;  
  delete[] inv_a_norm2;
}
__global__ void zUpdate(const float* __restrict__ A, const float* __restrict__ y, const float* __restrict__ x_half, const float* __restrict__ inv_a_norm2, float* z) {
  //int i = (blockDim.x*blockIdx.x + threadIdx.x) % n;
  //int j = (blockDim.x*blockIdx.x + threadIdx.x) / n;
  //int i_n_j = i + n*j;
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  float tmp = 0.0f;
  
  for (int j=0; j<d; j++)
    tmp += A[i + n*j]*(2*x_half[j] - z[i + n*j]);
    
  //z = z + x - x_half*ones(1,n);
  float beta = y[i]*fminf(fmaxf((1 - y[i]*tmp)*inv_a_norm2[i],0.0),alpha);
  for (int j=0; j<d; j++) 
    z[i + n*j] = x_half[j] + A[j+d*i]*beta;
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
