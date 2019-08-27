mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" Finito_async_parallel_mex.cpp

addpath('../../routine_work/');

Load_Data_and_Initialize;

disp('Start calculation');
tic

db_trained = Finito_async_parallel_mex(x, y, alpha, s, epoch, 16);

toc

fprintf('The cost is %.15f\n',f(db_trained', x, y, s));
fprintf('The decision boundary is %.15f, %.15f, %.15f, %.15f, ...\n', db_trained(1),db_trained(2),db_trained(3),db_trained(4));


load binary_test.mat

% test_x = [test_x ones(size(test_x, 1), 1)];

result = test_x * db_trained' > 0;
result = 2 * result - 1;
error_rate = 1 - sum(result == test_y) / size(result,1)