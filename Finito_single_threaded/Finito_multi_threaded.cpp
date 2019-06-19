#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <mex.h>
#include <thread>
#include <mutex>
#include <time.h>
#include <future>

using namespace std;

double sig (double x) {
  return 1/(1 + exp(-x));
}


double dot (double* x, double* y, int dim) {
  double result = 0;
  for (int i = 0; i < dim; i++) 
    result += x[i] * y[i];
  return result;
}

void vector_increment(double* x, double* incr, int dim){
  for (int i = 0; i < dim; i++)
    x[i] += incr[i];
}

double* grad_fi(double* w, double* xi, double yi, double s, int dim) {

  double* result = new double[dim];
    
  for (int j = 0; j < dim; j++) {
    result[j] = -sig( -yi * dot(w, xi, dim)) * yi * xi[j] + s * w[j];
  }
    
  return result;
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
    double s = 0;
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
    
  double** x_v = array2rowvectors (x, n, dim);  //delete

  double* zeros = new double [dim];
  for (int c = 0; c < dim; c++)
    zeros[c] = 0.0;
  
  double** z_v = new double* [n];
  for (int i = 0; i < n; i++) {
    z_v[i] = grad_fi(zeros, x_v[i], y[i], s, dim);
    for (int c = 0; c < dim; c++){
      z_v[i][c] = -1.0/alpha/s * z_v[i][c];
    }
  }
  
  delete[] zeros;
  
  double* mean_z =  mean_rowvectors(z_v, n, dim);

  int num_thread = 2;
  int num_itr = epoch * n;
  
  mutex sequential_mutex;
  mutex sync_mutex;
  mutex read_mutex;
  mutex* block_mutexes = new mutex[n];

  int thread_counter_sync = num_thread;
  int thread_counter_read = num_thread;

  auto safe_decrease_read = [&read_mutex](int& n) {
    lock_guard <mutex> lck(read_mutex);
    n -= 1;
  };

  auto safe_decrease_sync = [&sync_mutex](int &n) {
    lock_guard <mutex> lck(sync_mutex);
    n -= 1;
  };
  
  auto iterate = [&]() {
    
    while (num_itr > 0) {
      int ik = rand() % n;
      // mexPrintf("%d\n, ", ik);

      // Read mean_z
      double* old_mean_z = new double [dim];

      for (int i = 0; i < dim; i++) {
	old_mean_z[i] = mean_z[i];
      }
      safe_decrease_read(thread_counter_read);
      cout << "Itr: "<< num_itr << ", thread: " << this_thread::get_id() <<endl;
      // TODO: signal that read is done
   
      // Calculation
      double *grad_ik = grad_fi(old_mean_z, x_v[ik], y[ik], s, dim);
      
      for (int c =  0; c < dim; c++) {
	old_mean_z[c] -= 1.0/ alpha/ s * grad_ik[c];
      }
      // Now old_mean_z becomes new_z_ik
    
      double* incr_z = new double[dim];
      for (int c = 0; c < dim; c++)
	incr_z[c] = 1.0/n * (old_mean_z[c] - z_v[ik][c]);
      
      while (thread_counter_read > 0) {};
      // mean_z update
      vector_increment(mean_z, incr_z, dim);

      safe_decrease_sync(thread_counter_sync);
      while (thread_counter_sync);
      
      lock_guard <mutex> lck(sequential_mutex);
      // z_ik update
      delete[] z_v[ik];
      z_v[ik] = old_mean_z;

      delete[] incr_z;
      delete[] grad_ik;
      num_itr -= 1;

      // Here is a bug: You can't update counter before ALL thread finishes
      thread_counter_read = num_thread;
      thread_counter_sync = num_thread;
    }

  };
  srand(1);

  vector <thread> threads;

  for (int i = 0; i < num_thread; i++) {
    threads.push_back(thread(iterate));
  }

  for (auto& t: threads) t.join();


  // thread t1(iterate);
  // thread t2(iterate);
  // t1.join();
  // t2.join();
  // thread* threads = new thread[num_thread];

  // for (int t = 0; t < num_thread; t++){
  //   threads[t] = thread(iterate);
  // }

  // for (int t = 0; t < num_thread; t++){
  //   threads[t].join();
  // }


  
  // Output 
  
  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);
  double * ptr = mxGetPr(plhs[0]);
  
  for (int c = 0; c < dim; c++)
    ptr[c] = mean_z[c]; 
  
  delete[] mean_z;
  
  delete_double_ptr(x_v,n);
  delete_double_ptr(z_v,n);
  
  return;
  
}
