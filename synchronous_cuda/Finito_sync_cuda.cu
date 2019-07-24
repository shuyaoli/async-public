#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <curand.h> // CURAND_RNG_PSEUDO_MTGP32
#include <curand_kernel.h>
#include <chrono>
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

#define WARP_SIZE 32
#define n 8192
#define dim 1024
#define s 1
#define epoch 60
#define alpha 0.5

#define SIZE "LARGE"

#define NUM_PROCESSOR 2048 // > 1024
#define UPDATE_BLOCKSIZE 256// <=256

// zUpdate <<< NUM_PROCESSOR / UPDATE_BLOCKSIZE, UPDATE_BLOCKSIZE>>>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("Message: %s\n", cudaGetErrorString(x));\
    return EXIT_FAILURE;}} while(0)

using namespace std;

void read_var(double* var, string var_name, int len)
{
  string filename = string("../data/") + 
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

__global__ void initCurand (curandState *states, unsigned long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // possibly adding time to seq number 
  curand_init(seed, i, 0, &states[i]);
}

__global__ void reduction_sum_divided(const double* __restrict__ z,
                                     double* __restrict__ sum_z,
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
    temp = z[i + num_col * j];
    
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
      atomic_add(sum_z+j, temp);  
    // sum_z[j] += temp;
  }
}


__global__ void parallel_sum_divided(const double* __restrict__ z,
                                     double* __restrict__ sum_z,
                                     int num_col, double div) {
  // Lauch num_row threads in total
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // 1 ~ num_row
  double total = 0;
  for (int c = 0; c < num_col; c++) {
    total += z[idx * num_col + c];
  }
  sum_z[idx] = total / div;
}

__global__ void zUpdate(const double* __restrict__ x_a,
                        const double* __restrict__ y,
                        double* __restrict__ z_a,
                        double* __restrict__ mean_z,
                        double* __restrict__ delta_z,
                        curandState_t *states)
{

  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int ik =  curand (&states[idx]) % n;
  // int ik;
  // if (idx % 2 == 1)
  //   ik = idx;
  // else
  //   ik = 2 * idx;
  
  double dot = 0;
  for (int i = 0; i < dim; i++) 
    dot += mean_z[i] * x_a[dim * ik + i];

  for (int c =  0; c < dim; c++) {        
    delta_z[idx+c*NUM_PROCESSOR] = mean_z[c] - z_a[ik + c * n] - 
      alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_a[dim * ik + c] + s * mean_z[c]);
  }

  for (int c = 0; c < dim; c++) {
    // z_a[ik + c * n] += delta_z[idx + c * NUM_PROCESSOR];
    atomic_add(&z_a[ik + c * n], delta_z[idx + c * NUM_PROCESSOR]); // atomic gives < 0.1s performance loss
  }

  // ----UNCOMMENT this loop, then COMMENT OUT everything in the main loop except zUpdate
  // for (int c = 0; c < dim; c++){
  //   atomic_add(&mean_z[c], delta_z[idx + c * NUM_PROCESSOR] / n);
  // }
  //-----------------------------That's very bad, don't do it--------------------------------
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
  CUDA_CALL(cudaMalloc(&d_states, sizeof(curandState) * NUM_PROCESSOR));
  initCurand <<< NUM_PROCESSOR / 1024, 1024 >>> ( d_states, 0); //TODO: seed with time

  
  double* z_a =  new double[n * dim]();
  double* mean_z = new double [dim]();
  double *d_z_a, *d_mean_z;
  CUDA_CALL(cudaMalloc(&d_z_a, sizeof(double) * n * dim));
  CUDA_CALL(cudaMalloc(&d_mean_z, sizeof(double) * dim));                 
  CUDA_CALL(cudaMemcpy(d_z_a, z_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));

  double* d_delta_z;
  CUDA_CALL(cudaMalloc(&d_delta_z, sizeof(double) * dim * NUM_PROCESSOR));
  //----------SHOULD BE UNNECESSARY--------------------
  // double* delta_z = new double[NUM_THREAD * dim] ();
  // CUDA_CALL(cudaMemcpy(d_delta_z, delta_z, sizeof(double) * NUM_THREAD * dim, cudaMemcpyHostToDevice));
  //--------------------------------------------------

  double* delta_mean_z = new double [dim];
  double* d_delta_mean_z;
  CUDA_CALL(cudaMalloc(&d_delta_mean_z, sizeof(double) * dim));
  // CUDA_CALL(cudaMemcpy(d_delta_mean_z, delta_mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));
  chrono :: duration <double> elapsed (0);
  
  cudaDeviceSynchronize(); auto start = chrono :: high_resolution_clock::now();
  for (int k = 0; k < epoch * n / NUM_PROCESSOR ; k++) {
    // initCurand <<< NUM_THREAD / 1024, 1024 >>> ( d_states, k);
    
    zUpdate <<< NUM_PROCESSOR / UPDATE_BLOCKSIZE, UPDATE_BLOCKSIZE>>>
      (d_x_a, d_y, d_z_a, d_mean_z, d_delta_z, d_states);      // 2.6s
    
    //--------The following code enforce z_mean consistency somehow inefficiently-----------
    // memset(mean_z, 0, sizeof(double) * dim);
    // CUDA_CALL(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));  // < 0.01s
    // reduction_sum_divided <<< n / 1024, 1024>>> (d_z_a, d_mean_z, dim, n, n); // 0.35 s 
    //---------------------------------------------------------------

    //------------------------One way to calculate delta_mean_z-------------------------
    memset(delta_mean_z, 0, sizeof(double) * dim);
    CUDA_CALL(cudaMemcpy(d_delta_mean_z, delta_mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));
    // < 0.01 s
    reduction_sum_divided <<< NUM_PROCESSOR / 1024, 1024>>>
      (d_delta_z, d_delta_mean_z, dim, NUM_PROCESSOR, n); // 0.35 s

    //------------------Another way to calculate delta_mean_z----------------------------
    // parallel_sum_divided <<< dim / 1024, 1024 >>> (d_delta_z, d_delta_mean_z, NUM_PROCESSOR, n);
    //---------------------------------------------------------------------------------

    //---------------Comment out the following code when enforcing z_mean consistency------------
    
    CUDA_CALL(cudaMemcpy(delta_mean_z, d_delta_mean_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(mean_z, d_mean_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));
    
    for (int c = 0; c < dim; c++) {
      mean_z[c] += delta_mean_z[c];
    }
    
    CUDA_CALL(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));
    // < 0.01 s, including memory transfer time
    
    //-------------------------------------------------------------------------------------------
  }
  cudaDeviceSynchronize(); auto end = chrono::high_resolution_clock::now(); elapsed += end - start;
  cout << "elapsed time: " << elapsed.count() << " s\n";
  
  CUDA_CALL(cudaMemcpy(mean_z, d_mean_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));
  
  for (int i = 0; i < 4; i++) printf("%.15f\n", mean_z[i]);
  
  cudaFree(d_z_a);
  cudaFree(d_mean_z);
  cudaFree(d_x_a);
  cudaFree(d_y);
  cudaFree(d_delta_mean_z);
  cudaFree(d_delta_z);
  cudaFree(d_states);
  return 0;
}
