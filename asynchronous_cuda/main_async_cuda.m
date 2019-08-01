mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_75,code=\"sm_75,compute_75\"' Finito_async_cuda_mex.cu

addpath('../routine_work/');

Initialize;

tic
x_a = x';   
x_a = x_a(:);
x_a = gpuArray(x_a);
y = gpuArray(y);
toc

tic

NUM_AGENT = 1024;
BLOCKSIZE = 128;
db_trained = Finito_async_cuda_mex(x_a, y, alpha, s, epoch, NUM_AGENT, BLOCKSIZE);

toc


result = x * db_trained' > 0;
result = 2 * result - 1;

error_rate = 1 - sum(result == y) / size(result,1);
fprintf('The cost is %.15f\n',f(db_trained', x, y, s));
fprintf('The decision boundary is %.15f, %.15f, %.15f, %.15f\n', db_trained(1),db_trained(2),db_trained(3),db_trained(4));
rmpath('../routine_work/');