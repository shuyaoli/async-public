seed = 1;
err = 0.01; 
rng(seed, 'twister');

n = 8192; 
dim = 8192; 

alpha = 0.5;
epoch = 32;

s = 1; % regularizer

[x, y, ~] = generate_dataset(n, dim, err, seed); 