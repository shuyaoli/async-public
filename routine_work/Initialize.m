clear;
SAVE_GD_CUDA=1;

seed = 1;
err = 0.01; 
rng(seed);

n = 16384; 
dim = 7000; 

alpha = 0.5;
epoch = 30;

s = 1; % regularizer

[x, y, ~] = generate_dataset(n, dim, err, seed); 

if SAVE_GD_CUDA
    if n == 4096 && dim ==32
        x_a = x';
        x_a = x_a(:);
        writematrix(x_a, '../gd_cuda/SMALL/x_a');
        writematrix(y, '../gd_cuda/SMALL/y')
    end
    
    if n == 16384 && dim == 7000
        x_a = x';
        x_a = x_a(:);
        writematrix(x_a, '../gd_cuda/LARGE/x_a');
        writematrix(y, '../gd_cuda/LARGE/y')
    end

end