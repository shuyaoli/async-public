# include <iostream>
# include <cuda.h>
# define n 1024
# define dim 32
# include <cassert>
#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      printf("Message: %s\n", cudaGetErrorString(x));   \
      exit(1);}} while(0)

using namespace std;

int main()
{
  // Check whether Peer to peer access is available
  cudaSetDevice(0);
  int canAccessPeer;
  cudaError_t cudaResult;
  cudaResult = cudaDeviceCanAccessPeer(&canAccessPeer, 1, 0);
  cout << "Query Success: " << (cudaResult == cudaSuccess) << endl;
  cout << "Query Result: " << canAccessPeer << endl;

  // Get device property
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 1);
  cout << prop.canMapHostMemory << endl;

  // Portable memory (page-locked memory)
  double *d_z_a, *d_mean_z, *h_z_a, *h_mean_z;
  CUDA_CALL(cudaSetDevice(0));
  CUDA_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CALL(cudaHostAlloc(&h_z_a, sizeof(double) * n * dim, cudaHostAllocPortable & cudaHostAllocWriteCombined));
  CUDA_CALL(cudaHostAlloc(&h_mean_z, sizeof(double) * dim, cudaHostAllocPortable));
  
  if (cudaHostGetDevicePointer()) {
    CUDA_CALL(cudaHostGetDevicePointer(&h_z_a, d_z_a, 0));
    CUDA_CALL(cudaHostGetDevicePointer(&h_mean_z, d_mean_z, 0));
  }
 
  cudaFreeHost(h_z_a);
  cudaFreeHost(h_mean_z);
  
  unsigned int flags;
  cudaGetDeviceFlags(&flags);
  cout << &flags << endl;
  
  return 0;
}
