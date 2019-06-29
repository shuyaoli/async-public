mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" Finito_multi_threaded.cpp

addpath('../routine_work/');

Initialize;

tic

numthread = 8;

db_trained = Finito_multi_threaded(x, y, alpha, s, epoch, numthread);
% db_trained = Finito_multi_threaded_no_CAS(x, y, alpha, s, epoch, numthread);

toc


result = x * db_trained' > 0;
result = 2 * result - 1;

error_rate = 1 - sum(result == y) / size(result,1);
fprintf('The cost is %.15f\n',f(db_trained', x, y, s));

rmpath('../routine_work/');