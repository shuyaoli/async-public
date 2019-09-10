#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#define cudaDeviceMapHost 0x08

#define WARP_SIZE 32
#define n 8192
#define dim 1024
#define s 0.1 
#define epoch 64 
#define alpha 4

#define NUM_AGENT 512
#define BLOCKSIZE 128

#define SIZE "MEDIUM"

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
void read_var(double* ,string, int);
__global__ void run_async(const double* __restrict__ x_a,
                          const double* __restrict__ y,
                          double* z_a,
                          double* mean_z,
                          curandState* states,
                          int* itr_ptr,
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
      atomicAdd_system(itr_ptr, 1);
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
      // Intention: different warps can start from different coordinates to avoid collision
      // int d = (c + warpIdx * WARP_SIZE) % dim;
      // Doesn't work, basically no speed up.
      delta_buffer = mean_z[c] - z_a[ik * dim + c] - 
        alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_a[dim * ik + c] + s * mean_z[c]);
      atomicAdd_system(&z_a[ik * dim + c], delta_buffer);
      atomicAdd_system(&mean_z[c], delta_buffer / n);
    }

    
  }
}

int main()
{
  chrono :: duration <double> elapsed (0);
  chrono :: high_resolution_clock :: time_point start, end;
  
  double *x_a = new double [n * dim];
  double *y = new double [n];
  
  
  read_var(x_a, "x_a", n * dim);
  read_var(y, "y", n);

  // should reside on each GPUs
  cudaSetDevice(0);
  double *d_x0_a, *d_y0;
  CUDA_CALL(cudaMalloc(&d_x0_a, sizeof(double) * n * dim));
  CUDA_CALL(cudaMalloc(&d_y0, sizeof(double) * n ));
  CUDA_CALL(cudaMemcpy(d_x0_a, x_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_y0, y, sizeof(double) * n, cudaMemcpyHostToDevice));

  cudaSetDevice(1);
  double *d_x1_a, *d_y1;
  CUDA_CALL(cudaMalloc(&d_x1_a, sizeof(double) * n * dim));
  CUDA_CALL(cudaMalloc(&d_y1, sizeof(double) * n ));
  CUDA_CALL(cudaMemcpy(d_x1_a, x_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_y1, y, sizeof(double) * n, cudaMemcpyHostToDevice));
  
  // try to reside on host
  int *h_itr_ptr, *d_itr_ptr;
  CUDA_CALL(cudaHostAlloc(&h_itr_ptr, sizeof(int), cudaHostAllocMapped));
  CUDA_CALL(cudaHostGetDevicePointer(&h_itr_ptr, d_itr_ptr, 0));
  memset(h_itr_ptr, 0, sizeof(int));
  
  double *d_z_a, *d_mean_z, *h_z_a, *h_mean_z;
  CUDA_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CALL(cudaHostAlloc(&h_z_a, sizeof(double) * n * dim, cudaHostAllocMapped));
  CUDA_CALL(cudaHostAlloc(&h_mean_z, sizeof(double) * dim, cudaHostAllocMapped));  
  memset(h_z_a, 0, sizeof(double) * n * dim);
  memset(h_mean_z, 0, sizeof(double) * dim);
  CUDA_CALL(cudaHostGetDevicePointer(&h_z_a, d_z_a, 0));
  CUDA_CALL(cudaHostGetDevicePointer(&h_mean_z, d_mean_z, 0));

  // shouldn't matter, give each GPU a copy for it not to be a bottleneck
  cudaSetDevice(0);
  curandState *d_states0;
  CUDA_CALL(cudaMalloc(&d_states0, sizeof(curandState) * NUM_AGENT / 2)); // TODO: hardcoded

  cudaSetDevice(1);
  curandState *d_states1;
  CUDA_CALL(cudaMalloc(&d_states1, sizeof(curandState) * NUM_AGENT / 2));
  
  auto now_clock = time_point_cast<milliseconds>(system_clock::now());
  auto seed = now_clock.time_since_epoch().count();
  
  cudaDeviceSynchronize();
  start = high_resolution_clock::now();

  cudaSetDevice(0);
  run_async <<< NUM_AGENT * WARP_SIZE / BLOCKSIZE / 2, BLOCKSIZE>>> //TODO hardcoded
    (d_x0_a, d_y0, d_z_a, d_mean_z,  d_states0, d_itr_ptr, seed);

  cudaSetDevice(1);
  run_async <<< NUM_AGENT * WARP_SIZE / BLOCKSIZE / 2, BLOCKSIZE>>>
    (d_x1_a, d_y1, d_z_a, d_mean_z,  d_states1, d_itr_ptr, seed * 2);
  
  cudaDeviceSynchronize();
  end = chrono::high_resolution_clock::now();
  elapsed = end - start;
  
  for (int i = 0; i < 4; i++) printf("%.15f\n", d_mean_z[i]);
  
  cout <<"NUM_AGENT: " << NUM_AGENT << endl
       <<"BLOCKSIZE: " << BLOCKSIZE << endl
       <<"elapsed time: "<<elapsed.count()<<" s"<<endl
       << endl;
  
  cudaFree(d_itr_ptr);
  cudaFree(d_states0);
  cudaFree(d_states1);
  cudaFreeHost(h_z_a);
  cudaFreeHost(h_mean_z);
  cudaFree(d_x0_a);
  cudaFree(d_y0);
  cudaFree(d_x1_a);
  cudaFree(d_y1);
  delete []x_a;
  delete []y;
}



void read_var(double* var, string var_name, int len)
{
  string filename = string("../../data/") + 
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
