#include <iostream>
#include <utility>
#include <atomic>

using namespace std;

// Runtime check of whether the implementation of double is
// atomic-free
int main()
{
    std::atomic <double> a;

    std::cout << std::boolalpha
              << "std::atomic<A> is lock free? "
              << a.is_lock_free() << '\n';
}
