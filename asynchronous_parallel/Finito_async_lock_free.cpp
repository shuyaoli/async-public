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

// inline double sig (double x) noexcept {
//   return 1.0/(1 + exp(-x));
// }

inline double dot (double* x, double* y, int dim) noexcept {
  double result = 0.0;
  for (int i = 0; i < dim; i++) 
    result += x[i] * y[i];
  return result;
}

inline void atomic_double_fetch_add (atomic <double> &p, double a) noexcept {
  double old = p.load();
  double desired = old + a;
  while(!p.compare_exchange_weak(old, desired)) {
    desired = old + a;
  }
}

inline void atomic_vector_increment(atomic <double> x[], double incr[], int dim) noexcept{
  for (int i = 0; i < dim; i++)
    atomic_double_fetch_add (x[i], incr[i]);
}
// An suggested, possibly optimized version of cas;
// But for now I don't understand memory order

// void atomic_double_fetch_add (atomic <double> &p,
// 			      double a) {
//   double old = p.load(std::memory_order_consume);
//   double desired = old + a;
//   while(!p.compare_exchange_weak(old, desired,
//         std::memory_order_release, std::memory_order_consume)) {
//     desired = old + a;
//   }
// }

inline void vector_increment(double* x, double* incr, int dim) noexcept{
  for (int i = 0; i < dim; i++)
    x[i] += incr[i];
}

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

// inline void grad_fi(double* result, atomic <double>* w, double* xi, double yi, double s, int dim) noexcept {
//   for (int j = 0; j < dim; j++) {
//     result[j] = -1.0/(1+exp(yi * dot(w, xi, dim))) * yi * xi[j] + s * w[j].load();
//   }
// }

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

  atomic <double>** z_v = new atomic <double>* [n];
  for (int r = 0; r < n; r++) {
    z_v[r] = new atomic <double> [dim] ();
  }

  atomic_int itr_ctr(epoch * n); // Tracking iteration
  
  // Allocate shared memory for all threads
  atomic <double> *mean_z = new atomic <double> [dim] ();

  // mutex print_mutex;

  auto iterate = [&]() {
    // Allocate local memory for each thread
    // chrono :: duration <double> elapsed;
    
    double *delta_z = new double [dim];
    double *old_mean_z = new double [dim];
    double *old_z_ik = new double [dim];
    
    while (itr_ctr.load() > 0) {

      int ik = intRand(0, n - 1);

      // Read once - for optimizing purpose
      for (int c = 0; c < dim; c++) {
	old_mean_z[c] = mean_z[c].load();
	old_z_ik[c] = z_v[ik][c].load();
      }

      // Calculation for delta_z_ik
      double exptemp,utemp;
      for (int c = 0; c < dim; c++) {
	// auto start = chrono::high_resolution_clock::now();
	utemp = y[ik] * dot(old_mean_z, x_v[ik], dim);	
	// auto end = chrono::high_resolution_clock::now();
	// elapsed += end - start;
	exptemp = exp(utemp);
	delta_z[c] = -1.0 / (1+ exptemp) * y[ik] * x_v[ik][c] + s * old_mean_z[c];
      }

      for (int c =  0; c < dim; c++) {
    	delta_z[c] = old_mean_z[c] - old_z_ik[c] - 1.0/ alpha/ s * delta_z[c]; 
      }  // Now delta_z is delta_z_ik
  
      // update z_v[ik]
      for (int c = 0; c < dim; c++)
	atomic_double_fetch_add (z_v[ik][c], delta_z[c]);

      // increament mean_z
      for (int c = 0; c < dim; c++)
    	atomic_double_fetch_add (mean_z[c], delta_z[c]/n);
      
      // update iteration counter
      itr_ctr--;
    }
    
    // { // Timing
    //   lock_guard <mutex> lck(print_mutex);
    //   std::cout << "elapsed time: " << elapsed.count() << " s\n";
    //   // std::cout << "delta elapsed time: " << elapsed_delta.count() << " s\n";
    // }
    
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