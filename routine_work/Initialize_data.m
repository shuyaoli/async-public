seed = 1;
err = 0.01; 
rng(seed, 'twister');

n = 8192; 
dim = 1024; 

alpha = 4;
epoch = 64;

s = 0.1; % regularizer

[x, y, ~] = generate_dataset(n, dim, err, seed); 