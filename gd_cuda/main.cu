#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
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
#define epoch 30
#define alpha 0.5
#define SIZE "SMALL"
#define WARP_SIZE 32

using namespace std;

void err_chk(cudaError err) {
  if (err != cudaSuccess) {
    cout << cudaGetErrorString(err) << endl;
    assert(false);
  }
}

void read_var(double* var, string var_name, int len)
{
  string filename = string(SIZE) + string("/") + var_name + string(".txt");
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

__global__ void parallel_sum(const double* __restrict__ z, double *x_half) {
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
      buffer[threadIdx.x / WARP_SIZE] = temp;
    
    __syncthreads();


    // Add the remaining 32 partial sums using a single warp
    if(threadIdx.x < WARP_SIZE) {
      temp = buffer[threadIdx.x];
      for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
        temp += __shfl_xor_sync(0xffffffff,temp, delta);
    }

    // Add this block's sum to the total sum
    if(threadIdx.x == 0)
        atomic_add(x_half+j, temp);
      // x_half[j] = temp;
  }
}

int main()
{
  double *x_a = new double [n * dim];
  double *y = new double [n];

  read_var(x_a, "x_a", n * dim);
  read_var(y, "y", n);
  
  double* z_a =  new double[n * dim]();

  double* mean_z = new double [dim] ();

  // double *d_x_a, *d_y;
  double *d_z_a, *d_mean_z;

  // // err_chk(cudaMalloc((void**)&d_x_a, sizeof(double) * n * dim));
  // // err_chk(cudaMalloc((void**)&d_y, sizeof(double) * n ));
  err_chk(cudaMalloc((void**)&d_z_a, sizeof(double) * n * dim));
  err_chk(cudaMalloc((void**)&d_mean_z, sizeof(double) * dim));
  cout << "CUDA Malloc done\n";
  

  // // err_chk(cudaMemcpy(d_x_a, x_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  // // err_chk(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));

  // cout << "CUDA Memcpy Host to Device Done\n";

  
  for (int k = 0; k < epoch; k++) {
    
    for (int ik = 0; ik < n; ik++) {
      
      double dot = 0;
      for (int i = 0; i < dim; i++) 
        dot += mean_z[i] * x_a[dim * ik + i];

      for (int c =  0; c < dim; c++) {        
        z_a[ik+c*n] = mean_z[c] - alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_a[dim * ik + c] + s * mean_z[c]);
      }
    
    }
    err_chk(cudaMemcpy(d_z_a, z_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
    err_chk(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));

    parallel_sum <<< n / 1024, 1024>>> (d_z_a, d_mean_z);

    err_chk(cudaMemcpy(mean_z, d_mean_z, sizeof(double) * dim, cudaMemcpyDeviceToHost));
    
    // mean_rowvectors(mean_z, z_a, n, dim)
    // for (int c = 0; c < dim; c++) {
    //   double total = 0;
    //   for (int r = 0; r < n; r++) {
    //     total += z_a[r + c * n];
    //   }
    //   mean_z[c] = total / n;
    // }
    for (int c = 0; c < dim; c++)
      mean_z[c] /= n;
  }

  cudaFree(d_z_a);
  cudaFree(d_mean_z);
  
  for (int i = 0; i < dim; i++)
    printf("%.15f\n", mean_z[i]);
  
  return 0;
}
