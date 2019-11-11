This folder contains simple and abstract codes I wrote when I develop
more complex codes.

[GPU\_synchronization\_primitive](GPU_synchronization_primitive.cpp)
contains basic synchronization primitives (e.g. mutex, lock, read-write
lock) that are implemented with spin lock and more basic ones that are
supported on GPU. 

[atomic\_lock\_free\_Q](atomic_lock_free_Q.cpp) is used to query
whether certain atomic structure A ```atomic <A>``` is implemented by
compiler with lock or lock free, where A can be, say, double.
