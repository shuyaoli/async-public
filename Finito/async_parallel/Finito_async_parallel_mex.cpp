#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <time.h>
#include <condition_variable>
#include <atomic>
#include <chrono> 
#include <random>
#include <mex.h>

// #define num_T double
using namespace std;

// typedef double num_T;

inline int intRand(const int & min, const int & max) noexcept{
  static thread_local mt19937* generator = nullptr;
  if (!generator)
    generator = new mt19937(clock() +
                            hash<thread::id>()(this_thread::get_id()));
  uniform_int_distribution<int> distribution(min, max);
  return distribution(*generator);
}

inline void atomic_double_add (atomic <double> &p, double a) noexcept {
  double old = p.load();
  double desired;
  do {
    desired = old + a;
  } while(!p.compare_exchange_weak(old, desired));
}

// A suggested, possibly optimized version of cas;
// But for now I don't understand memory order

// void atomic_double_add (atomic <double> &p,
//                            double a) {
//   double old = p.load(std::memory_order_consume);
//   double desired = old + a;
//   while(!p.compare_exchange_weak(old, desired,
//         std::memory_order_release, std::memory_order_consume)) {
//     desired = old + a;
//   }
// }


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  //     Input: x, y, initial random weight phi, alpha, s, epoch
  //     Output: trained decision boundary

  const int n = mxGetDimensions(prhs[0])[0];
  const int dim = mxGetDimensions(prhs[0])[1];
    
  const double* x = mxGetPr(prhs[0]);
  const double* y = mxGetPr(prhs[1]);
  const double alpha = *mxGetPr(prhs[2]);
  const double s = *mxGetPr(prhs[3]);
  const int epoch = *mxGetPr(prhs[4]);
  const int num_thread = *mxGetPr(prhs[5]);
    
  double* x_a = new double [n * dim];
  for (int r = 0; r < n; r++)
    for (int c = 0; c < dim; c++)
      x_a[dim * r + c] = x[r + c * n];

  double* z_a = new double [n * dim] ();

  atomic_int itr_ctr(epoch * n); // Tracking iteration
  shared_mutex* block_mutex = new shared_mutex [n];
  // Allocate shared memory for all threads
  atomic <double> *mean_z = new atomic <double> [dim] ();

  mutex print_mutex;

  auto iterate = [&]() {
    // Allocate local memory for each thread
    double *delta_z = new double [dim];
    double *old_z_ik = new double [dim];
    
    while (itr_ctr.load() > 0) { // This loop takes 16.8s / 18.1s
      int ik = intRand(0, n - 1);

      block_mutex[ik].lock_shared();
      for (int c = 0; c < dim; c++)
        old_z_ik[c] = z_a[ik * dim + c];
      block_mutex[ik].unlock_shared();

      // Calculation for delta_z_ik
      //dot = <old_mean_z, x[ik]>, 1.0s / 18s
      double dot = 0;
      for (int c = 0; c < dim; c++) 
        dot += mean_z[c] * x_a[ik * dim + c];
     
      //delta_z = mean_z - z[ik] - alpha * grad_f[ik], 12s / 18s
      for (int c =  0; c < dim; c++) {
        delta_z[c] = mean_z[c] - old_z_ik[c]
          - alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_a[ik * dim + c] + s * mean_z[c]);
        // atomic_double_add(z_a[ik * dim + c], delta_buffer);
        // atomic_double_add(mean_z[c], delta_buffer/n);
      }
      // Now delta_z is delta_z_i

      // update z[ik], block lock faster than atomic variable
      // z[ik] += delta[z], 0.8s / 18s
      block_mutex[ik].lock();
      for (int c = 0; c < dim; c++)
        z_a[ik * dim + c] += delta_z[c];
      block_mutex[ik].unlock();
      
      // increment mean_z, 3.1 - 4.0 s
      // mean_z += delta_z / n
      for (int c = 0; c < dim; c++)
        atomic_double_add(mean_z[c], delta_z[c]/n);
      
      // update iteration counter
      itr_ctr--;
    }
    delete[] delta_z;
  };
  
  chrono :: duration <double > elapsed;
  auto start = chrono::high_resolution_clock::now();
  vector <thread> threads;
  for (int i = 0; i < num_thread; i++) {
    threads.push_back(thread(iterate));
  }
  for (auto& t: threads) t.join();
  auto end = chrono::high_resolution_clock::now();
  elapsed = (end - start);
  
  // MATLAB Output 
  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);
  double * ptr0 = mxGetPr(plhs[0]);
  for (int c = 0; c < dim; c++)
    ptr0[c] = mean_z[c].load();

  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *ptr1 = mxGetPr(plhs[1]);
  *ptr1 = elapsed.count();
  
  delete [] mean_z;
  delete [] x_a;
  delete [] z_a;
  delete [] block_mutex;
}
