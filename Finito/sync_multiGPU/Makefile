LIBS=-lcurand
LDFLAGS=-L/usr/local/cuda-10.0/lib64
main:
	nvcc -m64 -arch=sm_75 Finito_sync_cuda.cu -o main.o $(LIBS)
