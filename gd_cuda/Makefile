CPPFLAGS=-I//home/shuyao/MATLAB/R2019a/extern/include
LDFLAGS=-L/home/shuyao/MATLAB/R2019a/bin/glnxa64/
LIBS=-lmat -lmx
gd_cuda:
	nvcc $(CPPFLAGS) $(LDFLAGS) main.cu -o main.o $(LIBS)
