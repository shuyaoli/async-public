#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <curand.h> // CURAND_RNG_PSEUDO_MTGP32
#include <curand_kernel.h>
#include <chrono>

#define WARP_SIZE 32
#define n 8192
#define dim 1024
#define s 1
#define epoch 64
#define alpha 0.5

#define SIZE "MEDIUM"

#define NUM_PROCESSOR 4096    // > 1024
#define UPDATE_BLOCKSIZE 256  // <=256

// zUpdate <<< NUM_PROCESSOR / UPDATE_BLOCKSIZE, UPDATE_BLOCKSIZE>>>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("Message: %s\n", cudaGetErrorString(x));\
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("CuRand error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace std;

__device__ double atomic_add(double*, double);

//XXX is the __restrict__ keyword legitimate?
//const __restrict__ is valid for x_a and y since they are problem data so they never change
//__restrict__ for mean_z is valid since we do not change (write to) mean_z
//__restrict__ for delta_z is valid since the access is fully separated by indexing for this kernel
//__restrict__ for z_a is XXX not XX valid since (with the atomic writes) different threads can write
//to the same location. (Since there is only one read from z_a, I don't think it will make a difference
//but the __restrict__ seems to be conceptually wrong.)
__global__ void zCalculate(const double* __restrict__ x_a,
                           const double* __restrict__ y,
                           const double* __restrict__ z_a,
                           const double* __restrict__ mean_z,
                           double* __restrict__ delta_z,
                           const unsigned int* __restrict__ random_index,
                           int itr)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int ik =  random_index [itr * NUM_PROCESSOR + idx] % n;
  // Coalesced memory access is one of the code optimization
  // considerations in CUDA that actually matters. This has the
  // potential to greatly speed up or slow down your code. Currently,
  // the problem is that adjacent threads access different parts of
  // the dataset with the random number generation. One remedy is to
  // have a single warp access consecutive datapoints (circularly
  // consecutive, so use mod (%) to wrap around) by having only the
  // 0th thread in the warp generate a random index and sharing it
  // among the threads. Another option is to have the 32 threads
  // within a single warp process the same datapoint
  __shared__ double s_mean_z[dim];
  for (int c = 0; c < dim; c++)
    s_mean_z[c] = mean_z[c];
  
  //XXX Non-coalesced memory access XXX
  double dot = 0;
  for (int i = 0; i < dim; i++) 
    dot += s_mean_z[i] * x_a[dim * ik + i];

  //Coalesced memory access (good) for delta_z
  //XXX Non-coalesced memory access for z_a
  //XXX is this for-loop the main bottleneck? XXX
  //XXXthe read for mean_z could be shared across the block. consider using __shared__ variables
  //XXX or should could read mean_z[c] and share it across the warps using warp-level primitives
  for (int c =  0; c < dim; c++) {        
    delta_z[idx+c*NUM_PROCESSOR] = s_mean_z[c] - z_a[ik + c * n] - 
      alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_a[dim * ik + c] + s * s_mean_z[c]);
  }
}

__global__ void zUpdate(double* __restrict__ z_a,
                        double* __restrict__ delta_z,
                        unsigned int* __restrict__ random_index,
                        int itr)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int ik =  random_index [itr * NUM_PROCESSOR + idx] % n;
  //XXX Non-coalesced memory access XXX
  for (int c = 0; c < dim; c++) {
    // z_a[ik + c * n] += delta_z[idx + c * NUM_PROCESSOR];
    atomic_add(&z_a[ik + c * n], delta_z[idx + c * NUM_PROCESSOR]); // atomic gives < 0.1s performance loss
  }
}


void read_var(double* ,string, int);


__global__ void reduction_sum_divided(const double* __restrict__ z,
                                      double* __restrict__ sum_z,
                                      int num_row, int num_col, double div);


__global__ void parallel_sum_divided(const double* __restrict__ z,
                                     double* __restrict__ sum_z,
                                     int num_col, double div);

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
  CUDA_CALL(cudaMalloc(&d_delta_z, sizeof(double) * dim * NUM_PROCESSOR));

  double* delta_mean_z = new double [dim];
  double* d_delta_mean_z;
  CUDA_CALL(cudaMalloc(&d_delta_mean_z, sizeof(double) * dim));

  chrono :: duration <double> elapsed (0);

  unsigned int * d_random_index;
  CUDA_CALL(cudaMalloc(&d_random_index, sizeof(unsigned int) * n * epoch));
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL)); // seed
  CURAND_CALL(curandGenerate(gen, d_random_index, n * epoch));
  
  cudaDeviceSynchronize(); auto start = chrono :: high_resolution_clock::now();
  for (int k = 0; k < epoch * n / NUM_PROCESSOR ; k++) {

    zCalculate <<< NUM_PROCESSOR / UPDATE_BLOCKSIZE, UPDATE_BLOCKSIZE>>>
      (d_x_a, d_y, d_z_a, d_mean_z, d_delta_z, d_random_index, k);   // 2.6s

    zUpdate <<< NUM_PROCESSOR / UPDATE_BLOCKSIZE, UPDATE_BLOCKSIZE>>>
      (d_z_a, d_delta_z, d_random_index, k);
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
    //-------------------------------------------------------------------------------------------
    // < 0.01 s, including memory transfer time
    
    
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
