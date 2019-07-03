clear;
seed = 1;
rng(seed);

n = 500; 
dim = 3; 
err = 0.01; 

alpha = 1200;
epoch = 45;


%XXX what is s? XXX 
%XXX what is toMEX? XXX
s = 0; % function to be optimized, toMEX

[x, y, db] = generate_dataset(n, dim, err, seed); 
% x = 0.1 .* x;
% x y toMEX


rng('shuffle','twister');
phi = zeros(n,dim);

% phi = initializer_prefactor * 2 * (rand(n, dim) - 0.5); % n by dim, toMEX

% phi_bar : mean(phi, 1)
% phi_i = phi(i, :)
