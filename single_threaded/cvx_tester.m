cvx_begin
    variable w(dim)
    minimize f(w, x, y, s)
cvx_end

result = x * w > 0;
result = 2 * result - 1;

f(db_trained', x, y, s)
error_rate = 1 - sum(result == y) / size(result,1)