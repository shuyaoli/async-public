clear;
seed = 2;
rng(seed);

n = 1000;
dim = 6; % n > dim

x = randn(n,dim);
y = randn(n,1);

initializer_prefactor = 0;

s =  eigs(2 * x'* x, 1,'smallestabs');

alpha = 2;
epoch = 5000;

seed = 'shuffle';
rng(seed);
phi = initializer_prefactor * 2 * (rand(n, dim) - 0.5); % n by dim, toMEX
