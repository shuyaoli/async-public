Initialize;

tic

phi = Finito_single_threaded(x, y, phi, alpha, s, epoch);

toc

db_trained = mean(phi, 1) 

result = x * db_trained' > 0;
result = 2 * result - 1;

error_rate = 1 - sum(result == y) / size(result,1)
f(mean(phi, 1)', x, y, s)