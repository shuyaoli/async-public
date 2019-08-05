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
#define n 4096
#define dim 256
#define s 1
#define max_itr 16384
#define alpha 0.5

#define SIZE "MID"

#define NUM_AGENT 64

#define BLOCKSIZE 128


#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      printf("Message: %s\n", cudaGetErrorString(x));   \
      return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {    \
      printf("CuRand error at %s:%d\n",__FILE__,__LINE__);      \
      return EXIT_FAILURE;}} while(0)

using namespace std;
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

__global__ void zCalculate(const double* __restrict__ x_a,
                           const double* __restrict__ y,
                           const double* __restrict__ z,
                           double* delta_z,
                           const unsigned int* __restrict__ random_index,
                           int itr)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int lane = threadIdx.x % WARP_SIZE; // TODO: make sure that WARP_SIZE divides blockDim.x 
  const int warpIdx = idx / WARP_SIZE;
  const int ik =  random_index[itr * NUM_AGENT + warpIdx] % dim;

  double result = 0;

  for (int r = lane; r < n; r += WARP_SIZE) {
    double dot = 0;
    for (int c = 0; c < dim; c++)
      dot += z[c] * x_a[dim * r + c]; //TODO: shared across global memory
    result += -1.0 / (1+exp(y[r] * dot)) * y[r] * x_a[dim * r + ik] + s * z[ik];

  }
  __syncwarp();
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    result += __shfl_xor_sync(0xffffffff, result, offset);
  __syncwarp();

  if (lane == 0)
    delta_z[warpIdx] = result;  
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

  double *z = new double[dim] ();
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

  
  cudaDeviceSynchronize(); auto start = chrono :: high_resolution_clock::now();

  for (int k = 0; k < max_itr / NUM_AGENT; k++) {

    zCalculate <<< NUM_AGENT * WARP_SIZE / BLOCKSIZE, BLOCKSIZE>>>
      (d_x_a, d_y, d_z, d_delta_z, d_random_index, k);
  
    CUDA_CALL(cudaMemcpy(delta_z, d_delta_z, sizeof(double) * NUM_AGENT, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(z, d_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_AGENT; i++) {
      int ik = random_index[k * NUM_AGENT + i] % dim;
      z[ik] += -alpha * delta_z[i]/n;
    }
    
    CUDA_CALL(cudaMemcpy(d_z, z, sizeof(double) * dim, cudaMemcpyHostToDevice));    
  }

  cudaDeviceSynchronize(); auto end = chrono::high_resolution_clock::now(); elapsed += end - start;
  CUDA_CALL(cudaMemcpy(d_z, z, sizeof(double) * dim, cudaMemcpyHostToDevice));    


  for (int i = 0; i < 4; i++) printf("%.15f\n", z[i]);
  cout //<< endl
      <<"NUM_AGENT: " << NUM_AGENT << endl
      <<"BLOCKSIZE: " << BLOCKSIZE << endl
      <<"elapsed time: "<<elapsed.count()<<" s"<<endl
      << endl;
  
  CUDA_CALL(cudaMemcpy(z, d_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));

  cudaFree(d_x_a);
  cudaFree(d_y);
  cudaFree(d_delta_z);

  return 0;
}