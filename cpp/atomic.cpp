#include <iostream>
#include <utility>
#include <atomic>

using namespace std;


int main()
{
    std::atomic <double> a;

    std::cout << std::boolalpha
              << "std::atomic<A> is lock free? "
              << a.is_lock_free() << '\n';
              // << "std::atomic<B> is lock free? "
              // << std::atomic_is_lock_free(&b) << '\n';
}
