This folder contains simple and abstract codes I wrote when I develop
more complex codes.

[GPU\_synchronization\_primitive](GPU_synchronization_primitive.cpp)
contains basic synchronization primitives (e.g. mutex, lock, read-write
lock) that are implemented with spin lock and more basic ones that are
supported on GPU. 

[atomic\_lock\_free\_Q](atomic_lock_free_Q.cpp) is used to query
whether certain atomic structure ```std::atomic <A>``` is implemented
by compiler with lock or lock free, where A can be, say, double.

[multi\_GPU\_memory\_model](multi_GPU_memory_model.cu) is used to
check whether runtime GPU support various multi GPU memory models,
such as page-locked memory, peer memory access, etc.

In multithread\_for\_loop, different threads coordinate each other to
run a for loop simultaneously.
[multithread\_for\_loop-heavy\_work](multithread_for_loop-heavy_work.cpp)
works **only when** the for loop contains very heavy work but synchronizes
with one less lock and condition variable, while
[multithread\_for\_loop-all](multithread_for_loop-all.cpp) works for
all loops.

[reduction\_sum](reduction_sum.cu) utilizes GPU to sum up n arrays of
dimension d.
