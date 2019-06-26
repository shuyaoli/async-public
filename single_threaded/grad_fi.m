function out = grad_fi(w, xi, yi, s)
% out = grad_fi(w, xi, yi, s), w and xi are row vectors

out = -sig(- yi * dot(w, xi)) * yi * xi + s * w;

end

