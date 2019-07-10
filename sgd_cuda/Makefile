CPPFLAGS=-I/usr/local/MATLAB/R2016b/extern/include
LDFLAGS=-L/usr/local/MATLAB/R2016b/bin/glnxa64
LIBS=-lmat -lmx
sgd_cuda:
	nvcc $(CPPFLAGS) $(LDFLAGS) main.cpp -o main.o $(LIBS)
