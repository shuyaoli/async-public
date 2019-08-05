#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <time.h>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <random>
#include <mex.h>

using namespace std;

class Barrier {
private:
  int _ctr;
  mutex _mutex;
  condition_variable _cv;
public:
  Barrier (int = 0);
  void mod_incr(int);
  void sync ();
  void mod_incr_and_sync(int); // combine above two, but with one lock
};

Barrier::Barrier (int _init) {
  _ctr = _init;
}

void Barrier:: mod_incr(int _num_thread) {
  _mutex.lock();
  _ctr = (_ctr + 1) % _num_thread;
  _mutex.unlock();
}

void Barrier:: sync() {
  _mutex.lock();
  if (_ctr == 0) {
    _mutex.unlock();
    // Better to unlock before notifying, otherwise notified thread will wait for unlock
    _cv.notify_all();
  }
  else {
    _mutex.unlock();
    unique_lock <mutex> lck (_mutex);
    _cv.wait(lck, [this](){return _ctr == 0;});
    lck.unlock();
  }
}

void Barrier:: mod_incr_and_sync(int _num_thread) {
  _mutex.lock();
  _ctr = (_ctr + 1) % _num_thread;
  if (_ctr == 0) {
    _mutex.unlock();
    _cv.notify_all();
  }
  else {
    _mutex.unlock();
    unique_lock <mutex> lck (_mutex);
    _cv.wait(lck, [this](){return _ctr == 0;}); //strange capture pointer
    lck.unlock();
  }
}

inline int intRand(const int & min, const int & max) noexcept{
  static thread_local mt19937* generator = nullptr;
  if (!generator)
    generator = new mt19937( clock() + hash<thread::id>()(this_thread::get_id()) );
  uniform_int_distribution<int> distribution(min, max);
  return distribution(*generator);
}

inline void atomic_double_add (atomic <double> &p, double a) noexcept{
  double old = p.load();
  double desired;
  do {
    desired = old + a; // Can be inlined into next line without optimization
  } while(!p.compare_exchange_weak(old, desired));
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

void delete_nested_ptr (double **x, int M) {
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
  const int max_itr = *mxGetPr(prhs[4]);
  const int num_thread = *mxGetPr(prhs[5]);
  
  double** x_v = array2rowvectors (x, n, dim);  //new


  atomic_int itr_ctr(max_itr / num_thread * num_thread); // Tracking iteration
  
  Barrier read_barrier(0), itr_barrier(0);
  mutex *coord_mutex = new mutex [dim] ();
  mutex print_mutex; // For coordinating output of different threads.
  
  // Allocate shared memory for all threads
  // () at end is to "value-initialize", i.e., initialize all element to 0
  
  // atomic <double> *z = new atomic <double> [dim] ();
  double *z = new double [dim] ();
  
  // Prepare threads
  // iterate is a lambda expression with by-reference (&) capture mode
  // outside variables referenced in lambda body is accessed by reference
  auto iterate = [&]() {
    // Allocate local memory for each thread
    double *old_z = new double [dim];
    // chrono :: duration <double> elapsed;
    while (itr_ctr.load() > 0) {
      // update iteration counter
      itr_ctr--;
      
      itr_barrier.mod_incr_and_sync(num_thread); 

      /*****************************START*****************************/
      int ik = intRand(0, dim - 1);

      // Read mean_z, 0.54s / 25s
      for (int c = 0; c < dim; c++) {
        old_z[c] = z[c];
      }

      // Read is done
      read_barrier.mod_incr(num_thread); // < 0.01s
   
      double delta_z = 0;
      for (int r = 0; r < n; r++) {
        double dot = 0;
        for (int c = 0; c < dim; c++) 
          dot += old_z[c] * x_v[r][c];
        delta_z += -1.0 / (1+exp(y[r] * dot)) * y[r] * x_v[r][ik] + s * old_z[ik];
      }

      delta_z *= -alpha / n;
      

      /******************SYNCHRONIZATION*******************************/
      read_barrier.sync(); 
      /***************************************************************/
      coord_mutex[ik].lock();
      z[ik] += delta_z;
      coord_mutex[ik].unlock();
    }
    
    // print_mutex.lock();
    // std::cout << "C++ code elapsed time: " << elapsed.count() << " s\n";
    // print_mutex.unlock();

    delete[] old_z;
  };
  
  // Execute threads
  vector <thread> threads;
  chrono :: duration <double> elapsed (0);
  auto start = chrono :: high_resolution_clock::now();
  for (int i = 0; i < num_thread; i++)
    threads.push_back(thread(iterate));
  for (auto& t: threads) t.join();
  auto end = chrono::high_resolution_clock::now(); elapsed += end - start;
  cout << "high_resolution_clock elapsed time: " << elapsed.count() << " s\n";

  
  // MATLAB Output 
  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);
  double * ptr = mxGetPr(plhs[0]);
  for (int c = 0; c < dim; c++)
    ptr[c] = z[c]; 
  
  delete[] z;
  
  delete_nested_ptr(x_v,n);
}
