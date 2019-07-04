#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <time.h>
#include <condition_variable>
#include <atomic>
#include <chrono>  // debugging, set a waiting maximum
#include <random>
#include <mex.h>

using namespace std;

inline int intRand(const int & min, const int & max) noexcept{
  static thread_local mt19937* generator = nullptr;
  if (!generator)
    generator = new mt19937( clock() + hash<thread::id>()(this_thread::get_id()) );
  uniform_int_distribution<int> distribution(min, max);
  return distribution(*generator);
}

// This is not fetch add!
inline void atomic_double_fetch_add (atomic <double> &p, double a) noexcept{
  double old = p.load();
  double desired;
  do {
    desired = old + a; // Can be inlined into next line without optimization
  } while(!p.compare_exchange_weak(old, desired));
}

// An suggested, possibly optimized version of cas;
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

void atomic_vector_increment(atomic <double> x[], double incr[], int dim){
  for (int i = 0; i < dim; i++)
    atomic_double_fetch_add (x[i], incr[i]);
}

void vector_increment(double* x, double* incr, int dim){
  for (int i = 0; i < dim; i++)
    x[i] += incr[i];
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

  atomic_int sync_ctr(0), restart_ctr(0); // For synchronization
  atomic_int itr_ctr(epoch * n); // Tracking iteration
  atomic_int read_ctr(0);  // Make sure write is done after read

  // mutex print_mutex; // debugging purpose
  mutex sync_mutex; 
  mutex restart_mutex;
  mutex read_mutex;
  mutex* block_mutex = new mutex [n];
    
  condition_variable sync_cv, restart_cv, read_cv;
  
  // Allocate shared memory for all threads
  //() at end is to "value-initialize", i.e., initialize all element to 0
  atomic <double> *mean_z = new atomic <double> [dim] ();

  // Prepare threads
  // iterate is a lambda expression with by-reference (&) capture mode
  // outside variables referenced in lambda body is accessed by reference
  auto iterate = [&]() {
    // Allocate local memory for each thread
    double *old_mean_z = new double [dim];
    double *delta_z = new double [dim];
    
    while (itr_ctr.load() > 0) {

      // update iteration counter
      itr_ctr--;
      
      //XXX can this synchronozation be moved to the beginning of the whie loop? XXX
      //XXX Can this be done with a single object?
      //There are 3 parts to this synchronization
      // 1. Every thread waits for slowest thread
      // 2. Slowest thread executes some code
      //     ("some code" can be done with lambda expression with by-reference (&) capture mode)
      // 3. All threads resume
      //    These 3 could be made into 3 member function calls. XXX
      // The two synchronization features can be done with a single object.
      /****************************************************************/
      /**********************SYNCHRONIZATION***************************/
      /****************************************************************/
      sync_mutex.lock();

      sync_ctr.store((sync_ctr+1) % num_thread);
         
      //if statement executes by the slowest thread to arrive here
      if (sync_ctr.load()==0) {
        /****************************************************************/
        /*Whatever should be executed only once for each batch i/teration*/
        /***************************************************************/
        restart_ctr.store(num_thread - 1);
        read_ctr.store(0);
        /***************************************************************/
        notify_lck.unlock();
        sync_cv.notify_all();
      }

      else {
        sync_mutex.unlock();
      
        unique_lock <mutex> lck(sync_mutex);

        sync_cv.wait(lck, [&sync_ctr](){
                            return sync_ctr.load()==0; // do NOT lock before compare XXX what does this mean? XXX
                          });
    
        lck.unlock();
        restart_ctr--;
      }

      
      // XXX If we move the synchronization code at the of while loop
      // the code before "START" is unnecessary (I think) XXX
      // while(restart_ctr.load() > 0); //equivalent busy while loop
      /****************************************************************/
      /******************SYNCHRONIZATION*******************************/
      /****************************************************************/
      unique_lock <mutex> restart_notify_lck(restart_mutex);
      if (restart_ctr.load()==0) {
        restart_notify_lck.unlock();
        restart_cv.notify_all();
      }
      else
        restart_notify_lck.unlock();
      
      unique_lock <mutex> restart_lck(restart_mutex);
      restart_cv.wait(restart_lck, [&restart_ctr]{
          return restart_ctr.load() == 0;
        });
      restart_lck.unlock();
      /*****************************START*****************************/
      int ik = intRand(0, n - 1);

      // Read mean_z
      for (int c = 0; c < dim; c++) {
        old_mean_z[c] = mean_z[c].load();
      }
      read_ctr++;
      // Read is done
   
      // XXX what is s?? XXX
      // Calculation for delta_z_ik
      double dot = 0;
      
      //dot = old_mean_z^T * x_v
      for (int i = 0; i < dim; i++) 
        dot += old_mean_z[i] * x_v[ik][i];
      
      //delta_z = bar{z} - z_ikn - alpha/s ... XXX
      for (int c =  0; c < dim; c++) 
        delta_z[c] = old_mean_z[c] - z_v[ik][c] - 1.0/ alpha/ s * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_v[ik][c] + s * old_mean_z[c]);
      // Now delta_z is delta_z_ik

      //The lock is only meaningful when multiple threads pick the same index.
      //Should be rare when num_thread << n
      //compound statement lets lck go out of scope (destructor called) to release lock
       // update z_v[ik]
      block_mutex[ik].lock();
      for (int c = 0; c < dim; c++) {
        z_v[ik][c] += delta_z[c];
      }
      block_mutex[ik].unlock();
      

      for (int c = 0; c < dim; c++)
        delta_z[c] /= n;
      // Now delta_z is delta_mean_z
      
      /****************************************************************/
      /******************SYNCHRONIZATION*******************************/
      /***************************************************************/
      // while (read_ctr < num_thread); // equivalent busy while loop
      
      //XXX Can this be done with a single object?
      //There are 3 parts to this synchronization
      // 1. Set counter to 0
      // 2. atomically increment after read
      // 3. Wait until everything has completed read
      //    These 3 could be made into 3 member function calls. XXX
      unique_lock <mutex> read_notify_lck(read_mutex);
      if (read_ctr.load()==num_thread) {
        read_notify_lck.unlock();
        read_cv.notify_all();
      }
      else
        read_notify_lck.unlock();
        

      unique_lock <mutex> read_lck(read_mutex);
      read_cv.wait(read_lck, [&read_ctr, num_thread]{
          return read_ctr.load() == num_thread;
        });
      read_lck.unlock();
      /***************************************************************/

      // increament mean_z
      for (int c = 0; c < dim; c++) {
        atomic_double_fetch_add(mean_z[c], delta_z[c]);
      }


    }
    delete[] delta_z;
    delete[] old_mean_z;
  };

  // Execute threads
  vector <thread> threads;
  for (int i = 0; i < num_thread; i++)
    threads.push_back(thread(iterate));
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
