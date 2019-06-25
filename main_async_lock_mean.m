Initialize;

tic

db_trained = Finito_async_lock_mean(x, y, alpha, s, epoch, 8);
% db_trained = Finito_multi_threaded_no_CAS(x, y, alpha, s, epoch, 8);

toc


result = x * db_trained' > 0;
result = 2 * result - 1;

error_rate = 1 - sum(result == y) / size(result,1);
fprintf('The cost is %.15f\n',f(db_trained', x, y, s));