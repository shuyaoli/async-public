function out = f(w, x, y, s)
% out = f(w, x, y, s), sum of fi, NOT divided by n


% Is this separate function file necessary? where is f(...) used?

sigmoid = @(x) 1./(1 + exp(-x));

term1 = -log(sigmoid(y .* x * w));
term2 = s/2 * sum(w.^2);

out = sum(term1) / size(x, 1) + term2;

end

