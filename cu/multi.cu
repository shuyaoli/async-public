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
  cudaSetDevice(0);
  int canAccessPeer;
  cudaError_t cudaResult;
  cudaResult = cudaDeviceCanAccessPeer(&canAccessPeer, 1, 0);
  cout << "Query Success: " << (cudaResult == cudaSuccess) << endl;
  cout << "Query Result: " << canAccessPeer << endl;

  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, 1);
  // cout << prop.canMapHostMemory << endl;
  // double *d_z_a, *d_mean_z, *h_z_a, *h_mean_z;
  // CUDA_CALL(cudaSetDevice(0));
  // CUDA_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  // CUDA_CALL(cudaHostAlloc(&h_z_a, sizeof(double) * n * dim, cudaHostAllocPortable & cudaHostAllocWriteCombined));
  // CUDA_CALL(cudaHostAlloc(&h_mean_z, sizeof(double) * dim, cudaHostAllocPortable));
  // if (cudaHostGetDevicePointer()) {
  //   CUDA_CALL(cudaHostGetDevicePointer(&h_z_a, d_z_a, 0));
  //   CUDA_CALL(cudaHostGetDevicePointer(&h_mean_z, d_mean_z, 0));
  // }
 
  // cudaFreeHost(h_z_a);
  // cudaFreeHost(h_mean_z);
  // unsigned int flags;
  // cudaGetDeviceFlags(&flags);
  // cout << &flags << endl;
  return 0;
}

// #include <stdio.h>
// #include <assert.h>

// // Convenience function for checking CUDA runtime API results
// // can be wrapped around any runtime API call. No-op in release builds.
// inline
// cudaError_t checkCuda(cudaError_t result)
// {
// #if defined(DEBUG) || defined(_DEBUG)
//   if (result != cudaSuccess) {
//     fprintf(stderr, "CUDA Runtime Error: %s\n", 
//             cudaGetErrorString(result));
//     assert(result == cudaSuccess);
//   }
// #endif
//   return result;
// }

// void profileCopies(float        *h_a, 
//                    float        *h_b, 
//                    float        *d, 
//                    unsigned int  n,
//                    char         *desc)
// {
//   printf("\n%s transfers\n", desc);

//   unsigned int bytes = n * sizeof(float);

//   // events for timing
//   cudaEvent_t startEvent, stopEvent; 

//   checkCuda( cudaEventCreate(&startEvent) );
//   checkCuda( cudaEventCreate(&stopEvent) );

//   checkCuda( cudaEventRecord(startEvent, 0) );
//   checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
//   checkCuda( cudaEventRecord(stopEvent, 0) );
//   checkCuda( cudaEventSynchronize(stopEvent) );

//   float time;
//   checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
//   printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

//   checkCuda( cudaEventRecord(startEvent, 0) );
//   checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
//   checkCuda( cudaEventRecord(stopEvent, 0) );
//   checkCuda( cudaEventSynchronize(stopEvent) );

//   checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
//   printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

//   for (int i = 0; i < n; ++i) {
//     if (h_a[i] != h_b[i]) {
//       printf("*** %s transfers failed ***\n", desc);
//       break;
//     }
//   }

//   // clean up events
//   checkCuda( cudaEventDestroy(startEvent) );
//   checkCuda( cudaEventDestroy(stopEvent) );
// }

// int main()
// {
//   unsigned int nElements = 4*1024*1024;
//   const unsigned int bytes = nElements * sizeof(float);

//   // host arrays
//   float *h_aPageable, *h_bPageable;   
//   float *h_aPinned, *h_bPinned;

//   // device array
//   float *d_a;

//   // allocate and initialize
//   h_aPageable = (float*)malloc(bytes);                    // host pageable
//   h_bPageable = (float*)malloc(bytes);                    // host pageable
//   checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
//   checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
//   checkCuda( cudaMalloc((void**)&d_a, bytes) );           // device

//   for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;      
//   memcpy(h_aPinned, h_aPageable, bytes);
//   memset(h_bPageable, 0, bytes);
//   memset(h_bPinned, 0, bytes);

//   // output device info and transfer size
//   cudaDeviceProp prop;
//   checkCuda( cudaGetDeviceProperties(&prop, 0) );

//   printf("\nDevice: %s\n", prop.name);
//   printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

//   // perform copies and report bandwidth
//   profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
//   profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

//   printf("n");

//   // cleanup
//   cudaFree(d_a);
//   cudaFreeHost(h_aPinned);
//   cudaFreeHost(h_bPinned);
//   free(h_aPageable);
//   free(h_bPageable);

//   return 0;
// }