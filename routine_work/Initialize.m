clear;
seed = 1;
err = 0.01; 
rng(seed);

n = 5000; 
dim = 30; 

alpha = 0.5;
epoch = 30;

s = 1; % regularizer

[x, y, ~] = generate_dataset(n, dim, err, seed); 


