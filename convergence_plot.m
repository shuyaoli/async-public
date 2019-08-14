clear;
load('records.mat') % algorithm, n, dim, alpha, s, epoch, p, blocksize, times, function values

%% Enforce consistency of datasets
seed = 1;
err = 0.01; 
rng(seed, 'twister');

%% Begin experiments
expId = size(records,1) + 1;

n = 4096;
dim = 4096;

[x, y, ~] = generate_dataset(n, dim, err, seed); 

alpha = 1;
epoch = 64;

s = 0.1;
num_thread
records(1, 1:8) = {'finito', n, dim, alpha, s, epoch, 16