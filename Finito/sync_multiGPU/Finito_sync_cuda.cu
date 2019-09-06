#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <curand.h> 
#include <curand_kernel.h>
#include <chrono>

#define WARP_SIZE 32
#define n 8192
#define dim 1024
#define s 0.1
#define epoch 64
#define alpha 4

#define SIZE "MEDIUM"

#define NUM_AGENT 512

#define UPDATE_BLOCKSIZE 128
#define SUM_BLOCKSIZE 128
#define MEAN_BLOCKSIZE 128
// zCalculate <<< NUM_AGENT * WARP_SIZE / UPDATE_BLOCKSIZE, UPDATE_BLOCKSIZE>>>
// zUpdate    <<< NUM_AGENT * WARP_SIZE / UPDATE_BLOCKSIZE, UPDATE_BLOCKSIZE>>>
// parallel_sum_divided <<< dim / SUM_BLOCKSIZE, SUM_BLOCKSIZE>>> (d_delta_z, d_delta_mean_z, NUM_AGENT, dim, n);
// mean_zUpdate <<< dim / MEAN_BLOCKSIZE, MEAN_BLOCKSIZE >>> (d_delta_mean_z, d_mean_z);

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("Message: %s\n", cudaGetErrorString(x));\
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("CuRand error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace std;

// __device__ double atomic_add(double*, double);

// Legitimacy for the __restrict__ keyword
// const __restrict__ is valid for x_a and y since they are problem data so they never change
// const __restrict__ for mean_z and z_a is valid since we do not change (write to) mean_z and z_a
//__restrict__ for delta_z is valid since the access is fully separated by indexing for this kernel
__global__ void zCalculate(const double* __restrict__ x_a,
                           const double* __restrict__ y,
                           const double* __restrict__ z_a,
                           const double* __restrict__ mean_z,
                           double* __restrict__ delta_z,
                           const unsigned int* __restrict__ random_index,
                           int itr)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int lane = threadIdx.x % WARP_SIZE; // TODO: make sure that WARP_SIZE divides blockDim.x 
  const int warpIdx = idx / WARP_SIZE;
  const int ik =  random_index[itr * NUM_AGENT + warpIdx] % n;

  // Use of coalesced memory access in light of random index access
  
  // One option is to have a single warp access consecutive datapoints
  // (circularly consecutive, so use mod (%) to wrap around) by having
  // only the 0th thread in the warp generate a random index and
  // sharing it among the threads. (This option is implemented)

  // Another option is to have the 32 threads within a single warp
  // process the same datapoint
  
  // Correct. Do we need syncwarp?
  // The time spent is exactly the same as direct use of mean_z.
  // It seems that compiler did it for us
  // __shared__ double s_mean_z[dim];
  // for (int c = lane; c < dim; c+=WARP_SIZE)
  //   s_mean_z[c] = mean_z[c];
  // __syncwarp();

  // Correct. Slower than previous one.
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
                        int itr)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int lane = idx % WARP_SIZE; // TODO: threadIdx % WARP_SIZE
  const int warpIdx = idx / WARP_SIZE;
  const int ik =  random_index[itr * NUM_AGENT + warpIdx] % n;
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

void read_var(double* ,string, int);


__global__ void reduction_sum_divided(const double* __restrict__ z,
                                      double* __restrict__ sum_z,
                                      int num_row, int num_col, double div);


__global__ void parallel_sum_divided(const double* __restrict__ z,
                                     double* __restrict__ sum_z,
                                     int num_row, int num_col, double div);

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
  
  double* z_a =  new double[n * dim]();
  double* mean_z = new double [dim]();
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

  chrono :: duration <double> elapsed (0);

  unsigned int * d_random_index;
  CUDA_CALL(cudaMalloc(&d_random_index, sizeof(unsigned int) * n * epoch));
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL)); // TODO: seed with time
  CURAND_CALL(curandGenerate(gen, d_random_index, n * epoch));
  
  cudaDeviceSynchronize(); auto start = chrono :: high_resolution_clock::now();


  for (int k = 0; k < epoch * n / NUM_AGENT; k++) {
    memset(delta_mean_z, 0, sizeof(double) * dim);
    CUDA_CALL(cudaMemcpy(d_delta_mean_z, delta_mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));
    
    zCalculate <<< NUM_AGENT * WARP_SIZE / UPDATE_BLOCKSIZE, UPDATE_BLOCKSIZE>>>
      (d_x_a, d_y, d_z_a, d_mean_z, d_delta_z, d_random_index, k);
    
    zUpdate <<< NUM_AGENT * WARP_SIZE / UPDATE_BLOCKSIZE, UPDATE_BLOCKSIZE>>>
      (d_z_a, d_delta_z, d_random_index, k);

    parallel_sum_divided <<< 1 + (dim - 1) / SUM_BLOCKSIZE, SUM_BLOCKSIZE>>>
      (d_delta_z, d_delta_mean_z, NUM_AGENT, dim, n);
    //----------------------------------------------------------------------------

    mean_zUpdate <<< 1 + (dim - 1) / MEAN_BLOCKSIZE, MEAN_BLOCKSIZE>>>
      (d_delta_mean_z, d_mean_z, dim); 
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
  delete [] x_a;
  delete [] y;
  delete [] z_a;
  delete [] mean_z;
  delete [] delta_mean_z;
  return 0;
}

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

// __device__ double atomic_add(double* address, double val)
// {
//   unsigned long long int* address_as_ull =
//     (unsigned long long int*)address;
//   unsigned long long int old = *address_as_ull, assumed;

//   do {
//     assumed = old;
//     old = atomicCAS(address_as_ull, assumed,
//                     __double_as_longlong(val +
//                                          __longlong_as_double(assumed)));

//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//   } while (assumed != old);

//   return __longlong_as_double(old);
// }

// __global__ void reduction_sum_divided(const double* __restrict__ v,
//                                      double* __restrict__ sum_v,
//                                      int num_row, int num_col, double div) {
//   // Lauch num_col threads in total
  
//   // Holds intermediates in shared memory reduction
//   __syncthreads();
//   __shared__ double buffer[1024/WARP_SIZE];
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   int lane = threadIdx.x % WARP_SIZE;

//   for (int k = 0; k < num_row; k++) {
//     int j = (k + blockIdx.x) % num_row;
//     //j = k;
//     double temp;
//     // All threads in a block of 1024 take an element
//     temp = v[i + num_col * j];
    
//     // All warps in this block (32) compute the sum of all
//     // threads in their warp
//     for(int delta = WARP_SIZE/2; delta > 0; delta /= 2)
//       temp += __shfl_xor_sync(0xffffffff, temp, delta);

//     // Write all 32 of these partial sums to shared memory
//     if(lane == 0)
//       buffer[threadIdx.x / WARP_SIZE] = temp / div;
    
//     __syncthreads();

//     // Add the remaining 32 partial sums using a single warp
//     if(threadIdx.x < WARP_SIZE) {
//       temp = buffer[threadIdx.x];
//       for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
//         temp += __shfl_xor_sync(0xffffffff,temp, delta);
//     }

//     // Add this block's sum to the total sum
//     if(threadIdx.x == 0)
//       atomicAdd(sum_v+j, temp);  
//     // sum_v[j] += temp;
//   }
// }