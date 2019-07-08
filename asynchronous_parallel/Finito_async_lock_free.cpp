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

using namespace std;

inline int intRand(const int & min, const int & max) noexcept{
  static thread_local mt19937* generator = nullptr;
  if (!generator)
    generator = new mt19937(clock() +
                            hash<thread::id>()(this_thread::get_id()));
  uniform_int_distribution<int> distribution(min, max);
  return distribution(*generator);
}

inline void atomic_double_fetch_add (atomic <double> &p, double a) noexcept {
  double old = p.load();
  double desired;
  do {
    desired = old + a;
  } while(!p.compare_exchange_weak(old, desired));
}

// inline void atomic_vector_increment(atomic <double> x[], double incr[], int dim) noexcept{
//   for (int i = 0; i < dim; i++)
//     atomic_double_fetch_add (x[i], incr[i]);
// }

// A suggested, possibly optimized version of cas;
// But for now I don't understand memory order

// void atomic_double_fetch_add (atomic <double> &p,
//                            double a) {
//   double old = p.load(std::memory_order_consume);
//   double desired = old + a;
//   while(!p.compare_exchange_weak(old, desired,
//         std::memory_order_release, std::memory_order_consume)) {
//     desired = old + a;
//   }
// }


inline double** array2rowvectors (const double *array, int num_row, int num_col) noexcept {
  // return a vector of row vectors, e.g.
  //     input {1,2,3,4,5,6}, 
  //     output 
  //            1 3 5
  //            2 4 6
  double** matrix = new double* [num_row];
  for (int r = 0; r < num_row; r++) {
    matrix[r] = new double[num_col];
    for (int c = 0; c < num_col; c++) {
      matrix[r][c] = array[c * num_row + r];
    }
  }
  return matrix;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
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
    
  double** x_v = array2rowvectors (x, n, dim);  //new

  double** z_v = new double* [n];
  for (int r = 0; r < n; r++) {
    z_v[r] = new double [dim] ();
  }

  atomic_int itr_ctr(epoch * n); // Tracking iteration
  mutex* block_mutex = new mutex [n];
  // Allocate shared memory for all threads
  atomic <double> *mean_z = new atomic <double> [dim] ();

  mutex print_mutex;

  auto iterate = [&]() {
    // Allocate local memory for each thread
    
    chrono :: duration <double> elapsed;
    
    double *delta_z = new double [dim];
    
    while (itr_ctr.load() > 0) { // This loop takes 16.8s / 18.1s
      auto start = chrono::high_resolution_clock::now();
      
      int ik = intRand(0, n - 1);

      // Calculation for delta_z_ik

      //dot = <old_mean_z, x[ik]>, 1.0s / 18s

      double dot = 0;
      for (int i = 0; i < dim; i++) 
        dot += mean_z[i] * x_v[ik][i];
     
      //delta_z = mean_z - z[ik] - alpha * grad_f[ik], 12s / 18s
      
      for (int c =  0; c < dim; c++) 
        delta_z[c] = mean_z[c] - z_v[ik][c] - alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_v[ik][c] + s * mean_z[c]);

      // Now delta_z is delta_z_i


      // update z_v[ik]
      // z[ik] += delta[z], 0.8s / 18s

      block_mutex[ik].lock();
      for (int c = 0; c < dim; c++)
        z_v[ik][c] += delta_z[c];
      block_mutex[ik].unlock();
      

      // increment mean_z, 3.1 - 4.0 s
      // mean_z += delta_z / n
      for (int c = 0; c < dim; c++)
        atomic_double_fetch_add (mean_z[c], delta_z[c]/n);


      // update iteration counter
      
      itr_ctr--;
      auto end = chrono::high_resolution_clock::now();
      elapsed += end - start;

    }
    

    print_mutex.lock();
    std::cout << "elapsed time: " << elapsed.count() << " s\n";
    print_mutex.unlock();
    
    delete[] delta_z;
  };

  
  vector <thread> threads;
  for (int i = 0; i < num_thread; i++) {
    threads.push_back(thread(iterate));
  }
  for (auto& t: threads) t.join();

  
  // MATLAB Output 
  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);
  double * ptr = mxGetPr(plhs[0]);
  
  for (int c = 0; c < dim; c++)
    ptr[c] = mean_z[c].load();
  
  delete[] mean_z;
  
  for (int i = 0; i < n; i++)
    delete [] x_v[i];
  delete []x_v;

  for (int i = 0; i < n; i++)
    delete [] z_v[i];
  delete []z_v;
  
  // atomic <double> test;
  // cout << boolalpha
  //      <<"atomic <double>  is lock free? "
  //      << test.is_lock_free() << '\n';
}
