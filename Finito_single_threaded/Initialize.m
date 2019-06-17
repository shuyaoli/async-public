clear;
seed = 1;
rng(seed);

n = 1000; 
dim = 3; 
err = 0.01; 
initializer_prefactor = 1; 

alpha = 2;   % training, toMEX
epoch = 200; % training, toMEX

s = 1; % function to be optimized, toMEX

[x, y, db] = generate_dataset(n, dim, err, seed); 
% x = 0.1 .* x;
% x y toMEX

seed = 'shuffle';
rng(seed);
phi = initializer_prefactor * 2 * (rand(n, dim) - 0.5); % n by dim, toMEX

% phi_bar : mean(phi, 1)
% phi_i = phi(i, :)
