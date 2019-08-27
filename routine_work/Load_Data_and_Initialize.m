alpha = 1;
epoch = 128;

s = 1; 

load('binary_train.mat');

n = size(y, 1);

% x = [x ones(size(x,1), 1)];

dim = size(x, 2);