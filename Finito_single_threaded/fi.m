function out = fi(w, xi, yi, s)
% out = fi(w, xi, yi, s), w and xi are row vectors

out = -log(sig(yi * dot(xi, w))) + s/2 * norm(w,2) ^2; 

end

