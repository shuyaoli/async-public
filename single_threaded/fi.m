function out = fi(w, xi, yi, s)
% out = fi(w, xi, yi, s), w and xi are row vectors

sigmoid = @(x) 1./(1 + exp(-x));

out = -log(sigmoid(yi * dot(xi, w))) + s/2 * norm(w,2) ^2; 

end

