function out = grad_fi(w, xi, yi, s)
% out = grad_fi(w, xi, yi, s), w and xi are row vectors

sigmoid = @(x) 1./(1 + exp(-x));
out = -sigmoid(- yi * dot(w, xi)) * yi * xi + s * w;

end

