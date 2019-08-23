LIBS=-lcurand
LDFLAGS=-L/usr/local/cuda-10.0/lib64
recompile:
	nvcc -m64 -arch=sm_75 Finito_async_cuda.cu -o main.o $(LIBS)
