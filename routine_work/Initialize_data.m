seed = 1;
err = 0.01; 
rng(seed, 'twister');

n = 2048; 
dim = 16384; 

alpha = 4;
epoch = 128;

s = 0.1; % regularizer

[x, y, ~] = generate_dataset(n, dim, err, seed); 