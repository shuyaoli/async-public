clear;
seed = 1;
rng(seed);

n = 5000; 
dim = 300; 
err = 0.01; 

alpha = 2;
epoch = 30;

s = 1; % function to be optimized, toMEX

[x, y, db] = generate_dataset(n, dim, err, seed); 
% x = 0.1 .* x;
% x y toMEX

seed = 'shuffle';
rng(seed,'twister');
phi = zeros(n,dim);

% phi = initializer_prefactor * 2 * (rand(n, dim) - 0.5); % n by dim, toMEX

% phi_bar : mean(phi, 1)
% phi_i = phi(i, :)
