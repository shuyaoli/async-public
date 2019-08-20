clear;
cpath = pwd();
load('records.server.mat') % algorithm (3), n, dim, alpha, s, epoch, p, blocksize, times, function values
save('backup.mat', 'records')
addpath('routine_work');
disp('read records done')
%% Enforce consistency of datasets
seed = 1;
err = 0.01; 
rng(seed, 'twister');
%% Generate Data
for ctr = 1:3
    if ctr == 1
        n = 8192;
        dim = 8192;
    elseif ctr == 2
        n = 16384;
        dim = 2048;
    elseif ctr == 3
        n = 2048;
        dim = 16384;
    end

[x, y, ~] = generate_dataset(n, dim, err, seed); 

alphas = [0.1 1 4 16];
epoch = 32;
num_thread = 16;
s = 0.1;

num_agents = [128 256 512 1024 2048];
blocksize = 128;
COMP = 75;
disp('generate data done')
%% Begin experiments
% for alpha = alphas
%     for num_agent = num_agents
%         disp(sprintf("alpha = %d, num_agent = %d", alpha, num_agent));
%         records = finito_cuda_generate(records, x, y, n, dim, alpha, s, epoch, num_agent, blocksize, COMP, cpath);
%         records = scd_cuda_generate   (records, x, y, n, dim, alpha, s, epoch, num_agent, blocksize, COMP, cpath);
%         save('records.server.mat', 'records');
%     end
% end

for alpha = alphas
        disp(sprintf("alpha = %d", alpha));
        records = finito_parallel_generate(records, x, y, n, dim, alpha, s, epoch, num_thread, cpath);
        records = scd_parallel_generate(records, x, y, n, dim, alpha, s, epoch, num_thread, cpath);
        save('records.server.mat', 'records');
end
end
