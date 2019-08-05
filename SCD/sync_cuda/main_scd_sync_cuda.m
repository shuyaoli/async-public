mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  scd_sync_cuda_mex.cu

addpath('../../routine_work/');

Initialize;

 
x_a = x(:);
disp('Start calculation');

tic

NUM_AGENT = 256;
BLOCKSIZE = 256;
db_trained = scd_sync_cuda_mex(x_a, y, alpha, s, 40 * dim, NUM_AGENT, BLOCKSIZE, zeros(1, dim));

toc

result = x * db_trained' > 0;
result = 2 * result - 1;

error_rate = 1 - sum(result == y) / size(result,1);
fprintf('The cost is %.15f\n',f(db_trained', x, y, s));
fprintf('The decision boundary is %.15f, %.15f, %.15f, %.15f\n', db_trained(1),db_trained(2),db_trained(3),db_trained(4));
rmpath('../../routine_work/');