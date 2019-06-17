function out = grad_ls(w,x, y)

out = 2 * x' * (x * w - y);

end