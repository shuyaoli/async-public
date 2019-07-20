clear;
SAVE_CUDA_DATA=0;
if (SAVE_CUDA_DATA)
   input("Do you really want to save CUDA DATA? Press Ctrl-C to break. Input anything to continue");
end
seed = 1;
err = 0.01; 
rng(seed);

n = 8192; 
dim = 1024; 

alpha = 0.5;
epoch = 30;

s = 1; % regularizer

[x, y, ~] = generate_dataset(n, dim, err, seed); 

if SAVE_CUDA_DATA
    if n == 4096 && dim ==32
        x_a = x';
        x_a = x_a(:);
        writematrix(x_a, '../data/SMALL/x_a');
        writematrix(y, '../data/SMALL/y')
    end
    
    if n == 8192 && dim == 1024
        x_a = x';
        x_a = x_a(:);
        writematrix(x_a, '../data/LARGE/x_a');
        writematrix(y, '../data/LARGE/y')
    end

end