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

/** README for code timing**/


// For n = 10000, dim = 7000, epoch = 30, the code takes 17s - 35s to run (average ~25s)

// Documented time is the averaged time taken by each thread
using namespace std;

typedef double num_T;

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

inline void atomic_num_T_add (atomic <num_T> &p, num_T a) noexcept{
  num_T old = p.load();
  num_T desired;
  do {
    desired = old + a; // Can be inlined into next line without optimization
  } while(!p.compare_exchange_weak(old, desired));
}

// A suggested (from Reddit), possibly optimized version of cas;
// But for now I don't understand memory order

// void atomic_num_T_add (atomic <num_T> &p,
//                            num_T a) {
//   num_T old = p.load(std::memory_order_consume);
//   num_T desired = old + a;
//   while(!p.compare_exchange_weak(old, desired,
//         std::memory_order_release, std::memory_order_consume)) {
//     desired = old + a;
//   }
// }

num_T** array2rowvectors (const num_T *array, int num_row, int num_col) {
  // return a vector of row vectors
  // Test: 
  //     input {1,2,3,4,5,6}, 
  //     output 
  //            1 3 5
  //            2 4 6
  num_T** matrix = new num_T* [num_row];
  for (int r = 0; r < num_row; r++) {
    matrix[r] = new num_T[num_col];
    for (int c = 0; c < num_col; c++) {
      matrix[r][c] = array[c * num_row + r];
    }
  }
  return matrix;
}

void delete_nested_ptr (num_T **x, int M) {
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
    
  const num_T* x = mxGetPr(prhs[0]);
  const num_T* y = mxGetPr(prhs[1]);
  const num_T alpha = *mxGetPr(prhs[2]);
  const num_T s = *mxGetPr(prhs[3]);
  const int epoch = *mxGetPr(prhs[4]);
  const int num_thread = *mxGetPr(prhs[5]);
    
  num_T** x_v = array2rowvectors (x, n, dim);  //new

  num_T** z_v = new num_T* [n];
  for (int r = 0; r < n; r++) {
    z_v[r] = new num_T [dim] ();
  }

  atomic_int itr_ctr(epoch * n); // Tracking iteration
  
  Barrier read_barrier(0), itr_barrier(0);
  mutex* block_mutex = new mutex [n];
  mutex print_mutex; // For coordinating output of different threads.
  
  // Allocate shared memory for all threads
  //() at end is to "value-initialize", i.e., initialize all element to 0
  atomic <num_T> *mean_z = new atomic <num_T> [dim] ();

  // Prepare threads
  // iterate is a lambda expression with by-reference (&) capture mode
  // outside variables referenced in lambda body is accessed by reference
  auto iterate = [&]() {
    // Allocate local memory for each thread
    num_T *old_mean_z = new num_T [dim];
    num_T *delta_z = new num_T [dim];
    // chrono :: duration <num_T> elapsed;
    while (itr_ctr.load() > 0) { // This loop takes 23s/25s
      // update iteration counter
      itr_ctr--;

      itr_barrier.mod_incr_and_sync(num_thread); // 2.5s / 25s

      /*****************************START*****************************/
      int ik = intRand(0, n - 1);

      // Read mean_z, 0.54s / 25s
      for (int c = 0; c < dim; c++) {
        old_mean_z[c] = mean_z[c].load();
      }

      // Read is done

      read_barrier.mod_incr(num_thread); // < 0.01s
   
      //dot = <old_mean_z, x[ik]>, 1.3s / 25s
      num_T dot = 0;
      for (int i = 0; i < dim; i++) 
        dot += old_mean_z[i] * x_v[ik][i];

      
      //delta_z = mean_z - z[ik] - alpha * grad_f[ik], 11s / 25s
      for (int c =  0; c < dim; c++) 
        delta_z[c] = old_mean_z[c] - z_v[ik][c] - alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_v[ik][c] + s * old_mean_z[c]);

      // Now delta_z is delta_z_ik

      //The lock is only meaningful when multiple threads pick the same index.
      //Should be rare when num_thread << n

      // update z_v[ik], 0.84s / 24.8s
      // z[ik] += delta[z]
 
      block_mutex[ik].lock();
      for (int c = 0; c < dim; c++) {
        z_v[ik][c] += delta_z[c];
      }
      block_mutex[ik].unlock();

      // auto start = chrono::high_resolution_clock::now();
      /******************SYNCHRONIZATION*******************************/
      read_barrier.sync(); // 0.4s / 25s
      /***************************************************************/
      // auto end = chrono::high_resolution_clock::now();
      
      // increament mean_z, 5.5s / 25s
      // mean_z += delta_z / n
      for (int c = 0; c < dim; c++) {
        atomic_num_T_add(mean_z[c], delta_z[c] / n);
      }
      
      // elapsed += end - start;
    }
    
    // print_mutex.lock();
    // std::cout << "C++ code elapsed time: " << elapsed.count() << " s\n";
    // print_mutex.unlock();
    
    delete[] delta_z;
    delete[] old_mean_z;
  };

  // Execute threads
  vector <thread> threads;
  chrono :: duration <double> elapsed (0);
  auto start = chrono :: high_resolution_clock::now();
  for (int i = 0; i < num_thread; i++)
    threads.push_back(thread(iterate));
  for (auto& t: threads) t.join();
  auto end = chrono::high_resolution_clock::now(); elapsed += end - start;
  cout << "elapsed time: " << elapsed.count() << " s\n";
  // MATLAB Output 
  
  plhs[0] = mxCreateDoubleMatrix(1, dim, mxREAL);
  num_T * ptr = mxGetPr(plhs[0]);
  for (int c = 0; c < dim; c++)
    ptr[c] = mean_z[c].load(); 

  // plhs[1] = mxCreateNum_TMatrix(n, dim, mxREAL);
  // num_T * ptr1 = mxGetPr(plhs[1]);
  // for (int r = 0; r < n; r++)
  //   for (int c = 0; c < dim; c++)
  //     ptr1[r+c*n] = z_v[r][c];
  
  delete[] mean_z;
  
  delete_nested_ptr(x_v,n);
  delete_nested_ptr(z_v,n);
}
