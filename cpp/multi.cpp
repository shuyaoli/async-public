#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace std;

int main () {
  int num_thread = 3;
  int num_itr = 100;

  // atomic_bool restart(0);
  
  atomic_int thread_counter_sync(0);

  mutex sync_mutex, print_mutex, itr_mutex;
  
  condition_variable sync_cv;

  auto safe_decrease = [](int& n, mutex& mtx) {
    lock_guard <mutex> lck(mtx);
    // cout << "Decreased by "<< this_thread::get_id() <<endl;
    n -= 1;
  };

  auto safe_increase = [&](atomic_int& n, mutex& mtx) {
    print_mutex.lock();
    // cout << this_thread::get_id() << " wanted a lock for increase\n";
    print_mutex.unlock();
    lock_guard <mutex> lck(mtx);
    print_mutex.lock();
    cout << this_thread::get_id() << " obtain a lock for increase\n";
    print_mutex.unlock();
    n++;
    n.store(n.load() % num_thread);
  };

  auto safe_compare = [&](atomic_int& n, int val, mutex& mtx) {
    print_mutex.lock();
    // cout << this_thread::get_id() << " wanted a lock for compare\n";
    print_mutex.unlock();
    lock_guard <mutex> lck(mtx);
    print_mutex.lock();
    // cout << this_thread::get_id() << " obtain a lock for compare\n";
    print_mutex.unlock();
    return n.load() == val;
  };
  
  auto iterate = [&]() {

    while (num_itr > 0) {
      // restart.store(false);
      print_mutex.lock();cout << this_thread::get_id() << " start\n";print_mutex.unlock();

      safe_decrease(num_itr, itr_mutex);
 
      safe_increase(thread_counter_sync, sync_mutex);
      // notifying thread
      if (safe_compare(thread_counter_sync, 0, sync_mutex)) {
	
	print_mutex.lock();
	cout << this_thread::get_id() << " notify to wake up\n";
	print_mutex.unlock();

	sync_cv.notify_all();
      }
      // waiting thread
      else {
	
	sync_mutex.lock();
	print_mutex.lock();
	cout << this_thread::get_id() << " put to sleep at place " << thread_counter_sync.load() << endl;
	print_mutex.unlock();
	sync_mutex.unlock();

	unique_lock <mutex> lck(sync_mutex);

	
	sync_cv.wait(lck, [&](){	    
	    // sync_mutex.lock();
	    print_mutex.lock();
	    cout << this_thread::get_id()<< " wait status "<<(thread_counter_sync.load())<<endl;
	    print_mutex.unlock();
	    // sync_mutex.unlock();	    
	    return thread_counter_sync.load()==0; // do NOT lock before compare--locked before! dead lock!
	  });
	// lck.lock();
	cout << this_thread::get_id()<< " is resuming\n";
	lck.unlock();
	}
      
      
      //call_once to set the signal back
    }

  };
  srand(1);

  vector <thread> threads;

  for (int i = 0; i < num_thread; i++) {
    threads.push_back(thread(iterate));
  }

  for (auto& t: threads) t.join();

  return 0;
}
