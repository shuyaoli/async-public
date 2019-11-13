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
  mutex sync_mutex, print_mutex;
  condition_variable sync_cv;

  // Setup blueprint of parallel structure
  auto iterate = [&]() {
    while (true) {      
      { //debugging purpose
	lock_guard <mutex> lck(print_mutex);
	cout << this_thread::get_id() << " start\n";
      }
      
      std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
      // do some *heavy* work
      

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
        restart_ctr.store(num_thread - 1);
        { //debugging purpose 
          lock_guard <mutex> lck(print_mutex);
          cout << this_thread::get_id() <<" set restart_ctr to "
               << restart_ctr.load() <<endl;
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
			    // do NOT lock before compare
                            return thread_counter_sync.load()==0;
                          });
      
        lck.unlock();
        restart_ctr--;
        {
          lock_guard <mutex> lck(print_mutex);
          cout << this_thread::get_id() <<" dec restart_ctr to "
               << restart_ctr.load() <<endl;
        }
      }
    }};


  // Instantiate the blueprint and run
  vector <thread> threads;
  for (int i = 0; i < num_thread; i++) {
    threads.push_back(thread(iterate));
  }
  for (auto& t: threads) t.join();
  
  return 0;
}
