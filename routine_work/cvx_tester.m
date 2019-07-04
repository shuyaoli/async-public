addpath('../routine_work/');

Initialize;

cvx_begin
    variable w(dim)
    minimize f(w, x, y, s)
cvx_end

result = x * w > 0;
result = 2 * result - 1;

rmpath('../routine_work/');