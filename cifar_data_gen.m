clear;
cpath = pwd();
load('cifar.mat') % algorithm (3), n, dim, alpha, s, epoch, p, blocksize, times, function values
save('backup.mat', 'records')
addpath('routine_work');
disp('read records done')
%% load datasets
s = 1; 

load('binary_train.mat');

n = size(y, 1);

dim = size(x, 2);
%% Generate Data
alphas = [0.0625 0.125 0.25 1 4];
epoch = 32;
num_threads = [4 8 16 24 32];

num_agents = [16 32 64 128 256];
blocksizes = [128];
COMP = 75;
%% Begin experiments
for alpha = alphas
    for num_agent = num_agents
        for blocksize = blocksizes
            fprintf("alpha = %d, num_agent = %d, blosksize = %d\n", alpha, num_agent,blocksize);
            records = finito_cuda_generate(records, x, y, n, dim, alpha, s, epoch, num_agent, blocksize, COMP, cpath);
%             records = scd_cuda_generate   (records, x, y, n, dim, alpha, s, epoch, num_agent, blocksize, COMP, cpath);
            save('cifar.mat', 'records');
        end
    end
end

% for num_thread = num_threads
%     for alpha = alphas
%         fprintf("alpha = %d, num_agent = %d\n", alpha, num_thread);
%         records = finito_parallel_generate(records, x, y, n, dim, alpha, s, epoch, num_thread, cpath);
%         records = scd_parallel_generate(records, x, y, n, dim, alpha, s, epoch, num_thread, cpath);
%         save('cifar.mat', 'records');
%     end
% end