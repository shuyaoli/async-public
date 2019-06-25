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
#include <chrono>  // debugging, set a waiting maximum
#include <random>
#include <mex.h>

using namespace std;

int intRand(const int & min, const int & max) {
    static thread_local mt19937* generator = nullptr;
    if (!generator)
      generator = new mt19937(clock() +
			      hash<thread::id>()(this_thread::get_id()));
    uniform_int_distribution<int> distribution(min, max);
    return distribution(*generator);
}

double sig (double x) {
  return 1.0/(1 + exp(-x));
}

double dot (double* x, double* y, int dim) {
  double result = 0.0;
  for (int i = 0; i < dim; i++) 
    result += x[i] * y[i];
  return result;
}

void atomic_double_fetch_add (atomic <double> &p,
			      double a) {
  double old = p.load();
  double desired = old + a;
  while(!p.compare_exchange_weak(old, desired)) {
    desired = old + a;
  }
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

void atomic_vector_increment(atomic <double> x[], double incr[], int dim){
  for (int i = 0; i < dim; i++)
    atomic_double_fetch_add (x[i], incr[i]);
}

void vector_increment(double* x, double* incr, int dim){
  for (int i = 0; i < dim; i++)
    x[i] += incr[i];
}

void grad_fi(double* result, double* w, double* xi, double yi, double s, int dim) {
  for (int j = 0; j < dim; j++) {
    result[j] = -sig( -yi * dot(w, xi, dim)) * yi * xi[j] + s * w[j];
  }
}

double** array2rowvectors (const double *array, int num_row, int num_col) {
  // return a vector of row vectors
  // Test: 
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

void delete_double_ptr (double **x, int M) {
  for (int i = 0; i < M; i++)
    delete [] x[i];
  delete[]x;
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

  // atomic_int sync_ctr(0), restart_ctr(0); // For synchronization
  atomic_int itr_ctr(epoch * n); // Tracking iteration
  // atomic_int read_ctr(0);  // Make sure write is done after read

  // mutex print_mutex; // debugging purpose
  // mutex sync_mutex; 
  // mutex restart_mutex;
  // mutex read_mutex;
  shared_mutex mean_z_mutex;
  // unique_lock <mutex> mean_z_lock(mean_z_mutex, defer_lock);
  mutex* block_mutex = new mutex [n];
    
  // condition_variable sync_cv, restart_cv, read_cv;
  
  // Allocate shared memory for all threads
  atomic <double> *mean_z = new atomic <double> [dim] ();

  auto iterate = [&]() {
    // Allocate local memory for each thread
    double *old_mean_z = new double [dim];
    double *delta_z = new double [dim];
    
    while (itr_ctr.load() > 0) {

      int ik = intRand(0, n - 1);

      // Read mean_z
      {
        shared_lock lck(mean_z_mutex);
	for (int c = 0; c < dim; c++) {
	  old_mean_z[c] = mean_z[c].load();
	}
      }
      // Read is done
   
      // Calculation for delta_z_ik
      grad_fi(delta_z, old_mean_z, x_v[ik], y[ik], s, dim);
      for (int c =  0; c < dim; c++) {
	delta_z[c] = old_mean_z[c] - z_v[ik][c] - 1.0/ alpha/ s * delta_z[c]; 
      }  // Now delta_z is delta_z_ik

      { // update z_v[ik]
        lock_guard lck(block_mutex[ik]);
	vector_increment(z_v[ik], delta_z, dim);
      }

      for (int c = 0; c < dim; c++) {
	delta_z[c] /= n;
      } // Now delta_z is delta_mean_z
      

      // increament mean_z
      {
        unique_lock lck(mean_z_mutex);
	atomic_vector_increment(mean_z, delta_z, dim);
      }
      
      // update iteration counter
      itr_ctr--;
    }
    delete[] delta_z;
    delete[] old_mean_z;
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

  plhs[1] = mxCreateDoubleMatrix(n, dim, mxREAL);
  double * ptr1 = mxGetPr(plhs[1]);
  for (int r = 0; r < n; r++)
    for (int c = 0; c < dim; c++)
      ptr1[r+c*n] = z_v[r][c];
  
  delete[] mean_z;
  
  delete_double_ptr(x_v,n);
  delete_double_ptr(z_v,n);
}
