clear;
seed = 1;
rng(seed);

n = 10000; 
dim = 7000; 
err = 0.01; 

alpha = 0.5;
epoch = 30;


%XXX what is s? XXX 
%XXX what is toMEX? XXX
s = 1; % function to be optimized, toMEX

[x, y, db] = generate_dataset(n, dim, err, seed); 
% x = 0.1 .* x;
% x y toMEX


rng('shuffle','twister');
phi = zeros(n,dim);

% phi = initializer_prefactor * 2 * (rand(n, dim) - 0.5); % n by dim, toMEX

% phi_bar : mean(phi, 1)
% phi_i = phi(i, :)
