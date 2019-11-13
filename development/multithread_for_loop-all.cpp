#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

using namespace std;

int main () {
  int num_thread = 3;
  atomic_int thread_counter_sync(0), restart_ctr(0);
  mutex sync_mutex, print_mutex, restart_mutex;
  condition_variable sync_cv, restart_cv;

   // Setup blueprint of parallel structure
  auto iterate = [&]() {

    while (true) {

      { //debugging purpose
	lock_guard <mutex> lck(print_mutex);
	cout << this_thread::get_id() << " start loop\n";
      }
      
      sync_mutex.lock();
      int ctr = thread_counter_sync.load();
	
      { //debugging purpose
        lock_guard <mutex> lck(print_mutex);
        cout << this_thread::get_id() << " increase thread counter\n";
      }
	
      thread_counter_sync.store((ctr+1) % num_thread);
        
      if (thread_counter_sync.load()==0) {
        { //debugging purpose
          lock_guard <mutex> lck(print_mutex);
          cout << this_thread::get_id() << " notify to wake up\n";
        }

        sync_cv.notify_all();
        sync_mutex.unlock();
      }
      else {
        sync_mutex.unlock();
 
        { //debugging purpose
          lock_guard <mutex> lck(print_mutex);
          cout << this_thread::get_id() << " put to sleep at place "
               << thread_counter_sync.load() << endl;
        }

        unique_lock <mutex> lck(sync_mutex);

        sync_cv.wait(lck, [&](){	    
                            { //debugging purpose
                              lock_guard <mutex> lck(print_mutex);
                              cout << this_thread::get_id()<< " wait status "
                                   <<(thread_counter_sync.load())<<endl;
                            }   
                            return thread_counter_sync.load()==0; 
                          });
      
        lck.unlock();
        {
          lock_guard <mutex> lck(print_mutex);
          cout << this_thread::get_id() <<" dec restart_ctr to "
               << restart_ctr.load() <<endl;
        }
      }

      restart_mutex.lock();
      restart_ctr.store((restart_ctr + 1) % num_thread);
      restart_mutex.unlock();
      
      {
	lock_guard <mutex> lck(print_mutex);
	cout << this_thread::get_id() <<" set restart_ctr to "
	     << restart_ctr.load() <<endl;
      }

      restart_mutex.lock();
      if (restart_ctr.load()==0) {
        restart_mutex.unlock();
        restart_cv.notify_all();
      }
      else {
        restart_mutex.unlock();
      
      
        unique_lock <mutex> restart_lck(restart_mutex);
        restart_cv.wait(restart_lck, [&restart_ctr]{
				       return restart_ctr.load() == 0;
				     });
        restart_lck.unlock();
      }
      
      //work start here, heavy or not
      // std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
    }};


  // Instantiate the blueprint and run
  vector <thread> threads;
  for (int i = 0; i < num_thread; i++) {
    threads.push_back(thread(iterate));
  }
  for (auto& t: threads) t.join();
  
  return 0;
}
