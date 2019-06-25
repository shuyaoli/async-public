#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <time.h>
#include <future>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <random>

using namespace std;

void atomic_double_fetch_add (atomic <double> &p,
			      const double &a) {
  double old = p.load();
  double desired = old + a;
  while(!p.compare_exchange_weak(old, desired)) {
    desired = old + a;
  }
}

int main ()
{
  atomic <double> f (2.0);

  thread t1([](atomic <double> &fx) {
      double old = fx.load();

      double desired = old + 1;
      
      while(!fx.compare_exchange_weak(old, desired)) {
	this_thread::sleep_for (chrono::seconds(1));
	desired = old + 1;
      }
    }, ref(f));

  thread t2([](atomic <double> &fx) {
      fx.store(fx.load()+1);
    }, ref(f));

  t1.join();
  t2.join();


  cout << f <<endl;
  return 0;
}
