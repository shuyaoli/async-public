clear;
seed = 1;
rng(seed);

n = 10000; 
dim = 7000; 
err = 0.01; 

alpha = 0.5;
epoch = 30;


s = 1; % regularizer

[x, y, db] = generate_dataset(n, dim, err, seed); 


rng('shuffle','twister');
phi = zeros(n,dim);
