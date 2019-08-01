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
#define n 4000
#define dim 300
#define s 1 //
#define epoch 64 //
#define alpha 0.5 // 

#define NUM_AGENT 1024
#define BLOCKSIZE 128

#define SIZE "SMALL"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      printf("Message: %s\n", cudaGetErrorString(x));   \
      return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {    \
      printf("CuRand error at %s:%d\n",__FILE__,__LINE__);      \
      return EXIT_FAILURE;}} while(0)

using namespace std;

__global__ void run_async(const double* __restrict__ x_a,
                          const double* __restrict__ y,
                          double* z_a,
                          double* mean_z,
                          double* delta_z,
                          curandState* states,
                          int* itr_ptr)
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

__global__ void initCurand (curandState *states, unsigned long long seed)  {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, i, 0, &states[i]);
}

void read_var(double* ,string, int);


int main()
{
  chrono :: duration <double> elapsed (0);
  chrono :: high_resolution_clock :: time_point start, end;

  
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

  int zero = 0;
  int* d_itr_ptr;
  CUDA_CALL(cudaMalloc(&d_itr_ptr, sizeof(int)));
  CUDA_CALL(cudaMemcpy(d_itr_ptr, &zero, sizeof(int), cudaMemcpyHostToDevice));

  double* d_delta_z;
  curandState *d_states;
  
  CUDA_CALL(cudaMalloc(&d_delta_z, sizeof(double) * dim * NUM_AGENT));
  CUDA_CALL(cudaMalloc(&d_states, sizeof(curandState) * NUM_AGENT));
  
  cudaDeviceSynchronize();start=chrono::high_resolution_clock::now();
  run_async <<< NUM_AGENT * WARP_SIZE / BLOCKSIZE, BLOCKSIZE>>>
    (d_x_a, d_y, d_z_a, d_mean_z, d_delta_z, d_states, d_itr_ptr);
  cudaDeviceSynchronize();end=chrono::high_resolution_clock::now();elapsed=end-start;
  
  CUDA_CALL(cudaMemcpy(mean_z, d_mean_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 4; i++) printf("%.15f\n", mean_z[i]);
  
  cout<<"NUM_AGENT: " << NUM_AGENT << endl
      <<"BLOCKSIZE: " << BLOCKSIZE << endl
      <<"elapsed time: "<<elapsed.count()<<" s"<<endl
      << endl;
  cudaFree(d_delta_z);
  cudaFree(d_states);
  // Get memory ready for next run
  // memset(mean_z, 0, sizeof(double) * dim);
  // memset(z_a, 0, sizeof(double) * n * dim);
  // CUDA_CALL(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));
  // CUDA_CALL(cudaMemcpy(d_z_a, z_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  // CUDA_CALL(cudaMemcpy(d_itr_ptr, &zero, sizeof(int), cudaMemcpyHostToDevice)); 
  cudaFree(d_z_a);
  cudaFree(d_mean_z);
  cudaFree(d_x_a);
  cudaFree(d_y);
  delete []x_a;
  delete []y;
  delete []z_a;
  delete []mean_z;
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