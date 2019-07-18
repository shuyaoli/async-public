#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <curand.h> // CURAND_RNG_PSEUDO_MTGP32
#include <curand_kernel.h>
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

#define n 4096
#define dim 32
#define s 1
#define epoch 200
#define alpha 0.5
#define SIZE "SMALL"
#define WARP_SIZE 32
#define NUM_THREAD 8

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("Message: %s\n", cudaGetErrorString(x));\
    return EXIT_FAILURE;}} while(0)

using namespace std;

void err_chk(cudaError err) {
  if (err != cudaSuccess) {
    cout << cudaGetErrorString(err) << endl;
    assert(false);
  }
}

void read_var(double* var, string var_name, int len)
{
  string filename =
    string(SIZE) + string("/") + var_name + string(".txt");
  ifstream var_file(filename);
  string line;
  if (!var_file.is_open()) {
    cout << "Failed to open " << var_name << endl;
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < len; i++) {
    if (getline(var_file, line)) 
      var[i] = stod(line);
    else {
      cout << "Error loading " << var_name << endl;
      exit(EXIT_FAILURE);
    }
  }
  var_file.close();
}
// randomness design choice:
// different threads have different seeds; the same thread across different kernel lauches have the same seed but different sequence number
// 

__global__ void initCurand (curandState *states, unsigned long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // possibly adding time to seq number 
  curand_init(seed, i, 0, &states[i]);
}

__global__ void parallel_sum(const double* __restrict__ z,
                             double *sum_z) {
  //Holds intermediates in shared memory reduction
  __syncthreads();
  __shared__ double buffer[1024/WARP_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % WARP_SIZE;

  for (int k=0; k<dim; k++) {
    int j = (k + blockIdx.x) % dim;
    //j = k;
    double temp;
    // All threads in a block of 1024 take an element
    temp = z[i + n*j];
    
    // All warps in this block (32) compute the sum of all
    // threads in their warp
    for(int delta = WARP_SIZE/2; delta > 0; delta /= 2)
      temp += __shfl_xor_sync(0xffffffff, temp, delta);

    // Write all 32 of these partial sums to shared memory
    if(lane == 0)
      buffer[threadIdx.x / WARP_SIZE] = temp / n;
    
    __syncthreads();

    // Add the remaining 32 partial sums using a single warp
    if(threadIdx.x < WARP_SIZE) {
      temp = buffer[threadIdx.x];
      for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
        temp += __shfl_xor_sync(0xffffffff,temp, delta);
    }

    // Add this block's sum to the total sum
    if(threadIdx.x == 0)
      atomic_add(sum_z+j, temp);
    // sum_z[j] += temp;
  }
}

__global__ void zUpdate(const double* __restrict__ x_a,
                        const double* __restrict__ y,
                        double* z_a,
                        const double* __restrict__ mean_z,
                        double*  delta_z,
                        curandState_t *states)
{
  // const int ik = blockDim.x*blockIdx.x + threadIdx.x;
  const int idx = blockDim.x*blockIdx.x + threadIdx.x;
  const int ik =  curand (&states[idx]) % n;

  double dot = 0;
  for (int i = 0; i < dim; i++) 
    dot += mean_z[i] * x_a[dim * ik + i];

  for (int c =  0; c < dim; c++) {        
    delta_z[idx+c*NUM_THREAD] = mean_z[c] - z[ik + c * n] - 
      alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_a[dim * ik + c] + s * mean_z[c]);
  }

  // TODO: lock it!
  for (int c = 0; c < dim; c++) {
    z[ik + c * n] += delta_z[idx + c * NUM_THREAD];
  }
  
}

int main()
{
  double *x_a = new double [n * dim];
  double *y = new double [n];

  read_var(x_a, "x_a", n * dim);
  read_var(y, "y", n);
  
  double *d_x_a, *d_y;
  CUDA_CALL(cudaMalloc(&d_x_a, sizeof(double) * n * dim));
  CUDA_CALL(cudaMalloc(&d_y, sizeof(double) * n ));
  CUDA_CALL(cudaMemcpy(d_x_a, x_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));


  curandState *d_states;
  CUDA_CALL(cudaMalloc(&d_states, sizeof(curandState) * n));
  initCurand <<< n / 1024, 1024 >>> ( d_states, 0);

  
  double* z_a =  new double[n * dim]();
  double* mean_z = new double [dim]();
  double *d_z_a, *d_mean_z;
  CUDA_CALL(cudaMalloc(&d_z_a, sizeof(double) * n * dim));
  CUDA_CALL(cudaMalloc(&d_mean_z, sizeof(double) * dim));                 
  CUDA_CALL(cudaMemcpy(d_z_a, z_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));
  
  double* d_delta_z;
  CUDA_CALL(cudaMalloc(&d_delta_z, sizeof(double) * dim * n));
  

  for (int k = 0; k < epoch; k++) {
    // initCurand <<< n / 1024, 1024 >>> ( d_states, k);
    zUpdate <<< n / 1024, 1024 >>> (d_x_a, d_y, d_z_a, d_mean_z, d_delta_z, d_states);


    memset(mean_z, 0, sizeof(double) * dim);
    CUDA_CALL(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));

    
    parallel_sum <<< n / 1024, 1024>>> (d_z_a, d_mean_z);
  }

  CUDA_CALL(cudaMemcpy(mean_z, d_mean_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));
  
  for (int i = 0; i < dim; i++) printf("%.15f\n", mean_z[i]);
  
  cudaFree(d_z_a);
  cudaFree(d_mean_z);
  cudaFree(d_x_a);
  cudaFree(d_y);
  return 0;
}
