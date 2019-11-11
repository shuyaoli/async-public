# Asynchronous optimization algorithms with GPU
As the number of computational cores in CPUs and GPUs increases, so
does the cost of synchronization in parallel computing, and the need
for asynchronous algorithms grows. This project studies asynchronous
optimizations algorithms on GPUs.

GPUs have many more cores than CPUs do. However, the cores have small
local memory and are limited in their capability to communicate and
coordinate. Therefore, reducing the use of local memory and the amount
of communication and coordination is essential for designing an
asynchronous algorithm efficient on GPUs.

We extended the optimization algorithms [stochastic coordinate descent](https://arxiv.org/abs/1311.1873)
and [Finito](https://arxiv.org/abs/1407.2710) to the asynchronous setup and implemented them on CPUs and
GPUs. We observed that the asynchronous GPU algorithms were 20--30x
faster than the asynchronous CPU algorithms, which were 3.6--4x faster
than the synchronous parallel CPU algorithms.

# Code Structure

# Performance
![image alt >](/plot_8192_8192.jpg)

# Requirement

# Reference
