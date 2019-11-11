# Asynchronous Optimization Algorithms with GPU
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
Plot with Normal Scale     |  Semi-log(y) Plot
:-------------------------:|:-------------------------:
![](figure_for_README/plot_8192_8192.jpg)|![](figure_for_README/semilogy_8192_8192.jpg)
![](figure_for_README/plot_2048_16384.jpg)|![](figure_for_README/semilogy_2048_16384.jpg)
![](figure_for_README/plot_16384_2048.jpg)|![](figure_for_README/semilogy_16384_2048.jpg)

<!-- <img style="float: right;" src="figure_for_README/plot_8192_8192.jpg"/> -->
<!-- <img style="float: left;" src="figure_for_README/plot_2048_16384.jpg"/> -->
<!-- ![image alt >](/) -->
<!-- ![image alt <](/figure_for_README/plot_2048_16384.jpg) -->

# Requirement

# Reference
