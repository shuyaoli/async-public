addpath('../routine_work/');

Initialize;

tic

db_trained = Finito_single_threaded(x, y, alpha, s, epoch);

toc


result = x * db_trained' > 0;
result = 2 * result - 1;

error_rate = 1 - sum(result == y) / size(result,1);
fprintf('The cost is %.15f\n',f(db_trained', x, y, s));

rmpath('../routine_work/');