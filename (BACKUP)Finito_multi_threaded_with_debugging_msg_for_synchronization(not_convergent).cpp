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
  return 1/(1 + exp(-x));
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

  atomic_int sync_ctr(0), restart_ctr(0); // For synchronization
  atomic_int itr_ctr(epoch * n); // Tracking iteration
  atomic_int read_ctr(0);  // Make sure write is done after read
  bool* repeated_index = new bool [n] ();
  // mutex sequential_mutex;
  // mutex read_mutex;
  mutex print_mutex;
  mutex mean_z_mutex;
  
  mutex sync_mutex;
  mutex* block_mutex = new mutex [n];
    
  condition_variable sync_cv;
  
  auto iterate = [&]() {
    
    while (itr_ctr.load() > 0) {
      while (restart_ctr.load() != 0);
      // { //debugging purpose
      // 	lock_guard <mutex> lck(print_mutex);
      // 	cout << this_thread::get_id() << " start\n";
      // }
      
      // int ik = rand() % n; // mexPrintf("%d\n, ", ik);
      int ik = intRand(0, n - 1);
      // {
      // 	lock_guard<mutex> lk(print_mutex);
      // 	cout << ' ' << ik << ' ';
      // }
      //TODO: make sure randomness

      // Read mean_z
      double* old_mean_z = new double [dim]; // TODO: allocate once for each thread
      
      
      for (int c = 0; c < dim; c++) {
	old_mean_z[c] = mean_z[c];
      }
      
      read_ctr++;
      // Read is done
   
      // Calculation
      double *delta_z_ik = grad_fi(old_mean_z, x_v[ik], y[ik], s, dim);
    
      
      for (int c =  0; c < dim; c++) {
	delta_z_ik[c] *= - 1.0/ alpha/ s;
      }  // Now delta_z_ik stores the amount added to old_mean_z to become new_z_ik

      


      //------------------------------------------------------------------------------
      // SYNCHRONIZATION
      while (read_ctr < num_thread) {} //busy while loop
      //------------------------------------------------------------------------------
      // z_ik update with block_lock

      {
	lock_guard <mutex> lck(block_mutex[ik]);
	double* old_z_ik =  new double[dim];
	for (int c = 0; c < dim; c++)
	  old_z_ik[c] = z_v[ik][c];
	
	if (repeated_index[ik]) {
	  vector_increment(z_v[ik], delta_z_ik, dim);
	  delete[] old_mean_z;
	}
	else {
	  z_v[ik] = old_mean_z;
	  vector_increment(z_v[ik], delta_z_ik, dim);
	  repeated_index[ik] = 1;
	}
	
	double* incr_mean_z = new double[dim];
        for (int c = 0; c < dim; c++)
	  incr_mean_z[c] = 1.0/n * (z_v[ik][c]- old_z_ik[c]);
	{
	  lock_guard <mutex> lck(mean_z_mutex);
	  //mean_z += incr_z
	  //what if += is atomic? Then we don't have to lock (mutex)
	  //you can achieve atomic += with "compare-and-swap" or "compare-and-exchange"
	  vector_increment(mean_z, incr_mean_z, dim);       // mean_z update
	  delete[] incr_mean_z;
	}  
      // delete[] z_v[ik];
      // z_v[ik] = old_mean_z;   // TODO: a bug here - you should substract twice!
      }
    
      
      itr_ctr--;
      //------------------------------------------------------------------------------
      //SYNCHRONIZATION
      //------------------------------------------------------------------------------
      //TODO: Synchronize
      {
	lock_guard <mutex> lck(sync_mutex);
	int ctr = sync_ctr.load();
	
	// { //debugging purpose
	//   lock_guard <mutex> lck(print_mutex);
	//   cout << this_thread::get_id() << " obtain a lock for increase\n";
	// }
	
        sync_ctr.store((ctr+1) % num_thread);
	if (sync_ctr.load()==0) {
	  
	  // { //debugging purpose
	  //   lock_guard <mutex> lck(print_mutex);
	  //   cout << this_thread::get_id() << " notify to wake up\n";
	  // }
	  
	  /****************************************************************/
	  /*Whatever should be executed only once for each batch iteration*/
	  /***************************************************************/
	  
	  restart_ctr.store(num_thread - 1);
	  
	  read_ctr.store(0);
	  for (int i = 0; i < n; i++)
	    repeated_index[i] = 0;
	  
	  // cout << endl;
	  // { //debugging
	  //   lock_guard <mutex> lck(print_mutex);
	  //   cout << this_thread::get_id() <<" set restart_ctr to "
	  // 	 << restart_ctr.load() <<endl;
	  // }
	  sync_cv.notify_all();
	  continue;
	}
      }	  

      // { //debugging purpose
      // 	lock_guard <mutex> lck(print_mutex);
      // 	cout << this_thread::get_id() << " put to sleep at place "
      // 	<< sync_ctr.load() << endl;
      // }

      unique_lock <mutex> lck(sync_mutex);

      sync_cv.wait(lck, [&](){	    
	  // { //debugging purpose
	  //   lock_guard <mutex> lck(print_mutex);
	  //   cout << this_thread::get_id()<< " wait status "
	  // 	 <<(sync_ctr.load())<<endl;
	  // }   
	  return sync_ctr.load()==0; // do NOT lock before compare
	});
      
      lck.unlock();
      restart_ctr--;
      // { // debugging purpose
      // 	lock_guard <mutex> lck(print_mutex);
      // 	cout << this_thread::get_id() <<" dec restart_ctr to "
      // 	     << restart_ctr.load() <<endl;
      // }
    }};


  srand(1);

  vector <thread> threads;

  for (int i = 0; i < num_thread; i++) {
    threads.push_back(thread(iterate));
  }

  for (auto& t: threads) t.join();



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
