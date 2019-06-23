#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <mex.h>
#include <thread>
#include <mutex>
#include <time.h>
#include <future>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <random>

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

double* mean_rowvectors(double** x, int num_row, int num_col) {
  double* result = new double[num_col];
  for (int c = 0; c < num_col; c++) {
    double s = 0.0;
    for (int r = 0; r < num_row; r++)
      s += x[r][c];
    result[c] = s / num_row;
  }
  return result;
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
    
  double** x_v = array2rowvectors (x, n, dim);  //delete

  double** z_v = new double* [n];
  for (int i = 0; i < n; i++) {
    z_v[i] = new double [dim];
    for (int c = 0; c < dim; c++){
      z_v[i][c] = 1.0/alpha/s/2 * y[i] * x_v[i][c];
    }
  }

  atomic_int sync_ctr(0), restart_ctr(0); // For synchronization
  atomic_int itr_ctr(epoch * n); // Tracking iteration
  atomic_int read_ctr(0);  // Make sure write is done after read

  mutex print_mutex;
  mutex mean_z_mutex;
  mutex sync_mutex;
  mutex restart_mutex;
  mutex read_mutex;
  mutex* block_mutex = new mutex [n];
    
  condition_variable sync_cv, restart_cv, read_cv;
  
  // Allocate memory for all  threads
  double* mean_z =  mean_rowvectors(z_v, n, dim);
  bool* repeated_index = new bool [n] ();

  
  auto iterate = [&]() {
    // Allocate memory for each thread
    double *incr_mean_z = new double [dim];
    double *delta_z_ik = new double [dim];
    while (itr_ctr.load() > 0) {
      while(restart_ctr.load() > 0); //busy while loop
      // {
      // 	lock_guard <mutex> lck(restart_mutex);
      // 	if (restart_ctr.load()==0) {
      // 	  restart_cv.notify_all();
      // 	}
      // }
	  
      // unique_lock <mutex> restart_lck(restart_mutex);
      // restart_cv.wait(restart_lck, [&restart_ctr]{
      // 	  return restart_ctr.load() == 0;
      // 	});
      // restart_lck.unlock();
      
      // Allocate memory for each iteration
      int ik = intRand(0, n - 1);

      // Read mean_z
      double* old_mean_z = new double [dim]; 
      for (int c = 0; c < dim; c++) {
	old_mean_z[c] = mean_z[c];
      }
      read_ctr++;
      // Read is done
   
      // Calculation for delta_z_ik
      grad_fi(delta_z_ik, old_mean_z, x_v[ik], y[ik], s, dim);
      for (int c =  0; c < dim; c++) {
	delta_z_ik[c] *= - 1.0/ alpha/ s;
      }  // Now delta_z_ik stores the amount added to old_mean_z to become new_z_ik

      /****************************************************************/
      /******************SYNCHRONIZATION*******************************/
      /***************************************************************/
      // {
      // 	lock_guard <mutex> lck(read_mutex);
      // 	if (read_ctr.load()==num_thread) {
      // 	  read_cv.notify_all();
      // 	}
      // }
	  
      // unique_lock <mutex> read_lck(read_mutex);
      // read_cv.wait(read_lck, [&read_ctr, num_thread]{
      // 	  return read_ctr.load() == num_thread;
      // 	});
      // read_lck.unlock();
      while (read_ctr < num_thread) {} //busy while loop
      /***************************************************************/

      // z_ik update with block lock
      {
	lock_guard <mutex> lck(block_mutex[ik]);

	if (repeated_index[ik]) {
	  /****************************************************************/
	  /**********************Here is a bug!***************************/
	  /***************************************************************/
	  // vector_increment(z_v[ik], delta_z_ik, dim);
	  
	  // incr_mean_z = delta_z_ik;
	  // note that after mem optimization,
	  // both var are owned by thread, not iteration
	  
	  // for (int c = 0; c < dim; c++)
	  //   incr_mean_z[c] *= 1.0 / n;
	  /****************************************************************/
	  /**********************Bug Workaround***************************/
	  /*I did NOT update ANYTHING if index is repeated****************/
	  for (int c = 0; c < dim; c++)
	    incr_mean_z[c] = 0;
	  /****************************************************************/
	  delete[] old_mean_z;
	}
	else {
	  double* old_z_ik =  z_v[ik];

	  z_v[ik] = old_mean_z;
	  vector_increment(z_v[ik], delta_z_ik, dim);

	  for (int c = 0; c < dim; c++)
	    incr_mean_z[c] = 1.0/n * (z_v[ik][c]- old_z_ik[c]);

	  delete[] old_z_ik;

	  
	  repeated_index[ik] = 1;
	}

	{
	  lock_guard <mutex> lk(mean_z_mutex);
	  //mean_z += incr_z
	  //what if += is atomic? Then we don't have to lock (mutex)
	  //you can achieve atomic += with "compare-and-swap" or "compare-and-exchange"
	  vector_increment(mean_z, incr_mean_z, dim);       // mean_z update

	}  
      // deelte[] z_v[ik];
      // z_v[ik] = old_mean_z;   // TODO: a bug here - you should substract twice!
      }
    
      itr_ctr--;
      /****************************************************************/
      /**********************SYNCHRONIZATION***************************/
      /****************************************************************/
      {
	lock_guard <mutex> lck(sync_mutex);
	int ctr = sync_ctr.load();
	sync_ctr.store((ctr+1) % num_thread);
	if (sync_ctr.load()==0) {
	  /****************************************************************/
	  /*Whatever should be executed only once for each batch iteration*/
	  /***************************************************************/
	  restart_ctr.store(num_thread - 1);
	  read_ctr.store(0);

	  /***************************************************************/
	  sync_cv.notify_all();
	  repeated_index[ik] = 0;
	  continue;
	}
      }
      
      unique_lock <mutex> lck(sync_mutex);

      sync_cv.wait(lck, [&sync_ctr](){
	  return sync_ctr.load()==0; // do NOT lock before compare
	});
      
      lck.unlock();
      repeated_index[ik] = 0;
      restart_ctr--;
    }
    delete[] incr_mean_z;
    delete[] delta_z_ik;
  };

  vector <thread> threads;

  for (int i = 0; i < num_thread; i++) {
    threads.push_back(thread(iterate));
  }

  for (auto& t: threads) t.join();

  // Output 
  
  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);
  double * ptr = mxGetPr(plhs[0]);
  
  for (int c = 0; c < dim; c++)
    ptr[c] = mean_z[c]; 

  plhs[1] = mxCreateDoubleMatrix(n, dim, mxREAL);
  double * ptr1 = mxGetPr(plhs[1]);
  for (int r = 0; r < n; r++)
    for (int c = 0; c < dim; c++)
      ptr1[r+c*n] = z_v[r][c];
  
  delete[] mean_z;
  
  delete_double_ptr(x_v,n);
  delete_double_ptr(z_v,n);
  
  return;
  
}
