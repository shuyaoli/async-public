clear;
seed = 1;
rng(seed);

n = 5000; 
dim = 300; 
err = 0.01; 

alpha = 0.5;
epoch = 30;


s = 1; % regularizer

[x, y, db] = generate_dataset(n, dim, err, seed); 


rng('shuffle','twister');
phi = zeros(n,dim);
