#include <atomic>
#include <iostream>
#include <thread>
#include <mutex>
using namespace std;

class CudaMutex {
private:
  atomic <int> _m;
public:
  void init () {_m = 0;};
   void lock (){
     while ( _m.exchange(1) == 1) {}
  };
   void unlock() {_m = 0;};
};

class ReaderWriterLock {
private:
  CudaMutex ctr_mutex;
  CudaMutex global_mutex;
  int b;
public:
  ReaderWriterLock () {
    ctr_mutex.init();
    global_mutex.init();
    b = 0;
  };  
   void read_lock() {
    ctr_mutex.lock();
    b++;
    if (b == 1)
      global_mutex.lock();
    ctr_mutex.unlock();
  };
   void read_unlock() {
    ctr_mutex.lock();
    b--;
    if (b == 0)
      global_mutex.unlock();
    ctr_mutex.unlock();
  };
   void write_lock(){
    global_mutex.lock();
  };
   void write_unlock(){
    global_mutex.unlock();
  };
};



class ThreadSafeCounter {
 public:
  ThreadSafeCounter() = default;
 
  // Multiple threads/readers can read the counter's value at the same time.
  void get(unsigned int *output) {
    mutex_.read_lock();
    *output = value_;
    mutex_.read_unlock();
  }
 
  // Only one thread/writer can increment/write the counter's value.
  void increment() {
    mutex_.write_lock();
    value_++;
    mutex_.write_unlock();
  }
 
  // Only one thread/writer can reset/write the counter's value.
  void reset() {
    mutex_.write_lock();
    value_ = 0;
    mutex_.write_unlock();
  }
 
 private:
   ReaderWriterLock mutex_;
  unsigned int value_ = 0;
};
 
int main() {
  ThreadSafeCounter counter;
  CudaMutex print_mutex;
  
  auto increment_and_print = [&]() {
    for (int i = 0; i < 3; i++) {
      counter.increment();
      unsigned int output;
      counter.get(&output);

      
      print_mutex.lock();
      std::cout << std::this_thread::get_id() << ' ' << output << '\n';
      print_mutex.unlock();
      // Note: Writing to std::cout actually needs to be synchronized as well
      // by another std::mutex. This has been omitted to keep the example small.
    }
  };
 
  std::thread thread1(increment_and_print);
  std::thread thread2(increment_and_print);
 
  thread1.join();
  thread2.join();
}
// {
//   atomic <double> f (2.0);

//   thread t1([](atomic <double> &fx) {
//       double old = fx.load();

//       double desired = old + 1;
      
//       while(!fx.compare_exchange_weak(old, desired)) {
// 	this_thread::sleep_for (chrono::seconds(1));
// 	desired = old + 1;
//       }
//     }, ref(f));

//   thread t2([](atomic <double> &fx) {
//       fx.store(fx.load()+1);
//     }, ref(f));

//   t1.join();
//   t2.join();


//   cout << f <<endl;
//   return 0;
// }
