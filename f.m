function out = f(w, x, y, s)
% out = f(w, x, y, s), sum of fi, NOT divided by n

term1 = -log(sig(y .* x * w));
term2 = s/2 * sum(w.^2);

out = sum(term1) / size(x, 1) + term2;

end

