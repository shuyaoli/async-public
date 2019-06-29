function [input, output, db] = generate_dataset(n, dim, err, seed)
%[input,output, decision_boundary] = generate_dataset(n, dim, error)
%   n is the number of data points
%   dim is the length of each data point
%   err is the noise level for classification (error rate)
%   input is an n by dim matrix as input data
%   output is a n-dim vector as output labels from {+1, -1}
%   db is the vector representing decision boundary
%   seed controls the random number generator


rng(seed)
db = 2 * rand(dim,1) - 1;  % ~Unif[-1,1]
input = 2 * rand(n,dim) - 1;  % ~Unif[-1,1]
output = input * db > 0;
flip = rand(n, 1) < err;
output = output + flip;
output = rem(output, 2);   % output mod 2
output = 2 * output - 1;

end

