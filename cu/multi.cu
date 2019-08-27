# include <iostream>
# include <cuda.h>

using namespace std;

int main()
{
  cudaSetDevice(0);
  int canAccessPeer;
  cudaError_t cudaResult;
  cudaResult = cudaDeviceCanAccessPeer(&canAccessPeer, 1, 0);
  cout << "Query Success: " << (cudaResult == cudaSuccess) << endl;
  cout << "Query Result: " << canAccessPeer << endl;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 3);
  cout << prop.unifiedAddressing << endl;
  return 0;
}