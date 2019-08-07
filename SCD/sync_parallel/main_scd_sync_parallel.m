mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" scd_sync_parallel.cpp

addpath('../../routine_work/');

Initialize;

tic

numthread = 16;

db_trained = scd_sync_parallel(x, y, alpha, s, epoch * dim, numthread);

toc


result = x * db_trained' > 0;
result = 2 * result - 1;

error_rate = 1 - sum(result == y) / size(result,1);
fprintf('The cost is %.15f\n',f(db_trained', x, y, s));
fprintf('The decision boundary is %.15f, %.15f, %.15f, %.15f\n', db_trained(1),db_trained(2),db_trained(3),db_trained(4));
