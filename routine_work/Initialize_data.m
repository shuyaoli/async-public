seed = 1;
err = 0.01; 
rng(seed, 'twister');

n = 4096; 
dim = 4096; 

alpha = 2.5;
epoch = 64;

s = 0.1; % regularizer

[x, y, ~] = generate_dataset(n, dim, err, seed); 