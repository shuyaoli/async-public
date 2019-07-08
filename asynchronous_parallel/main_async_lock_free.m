mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" Finito_async_lock_free.cpp

addpath('../routine_work/');

Initialize;

tic

db_trained = Finito_async_lock_free(x, y, alpha, s, epoch, 8);

toc
% print

result = x * db_trained' > 0;
result = 2 * result - 1;

error_rate = 1 - sum(result == y) / size(result,1);
fprintf('The cost is %.15f\n',f(db_trained', x, y, s));

rmpath('../routine_work/');