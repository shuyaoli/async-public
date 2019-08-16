clear;
cpath = pwd();
load('records.mat') % algorithm (3), n, dim, alpha, s, epoch, p, blocksize, times, function values
save('records.mat.backup', 'records')
addpath('routine_work');
disp('read records done')
%% Enforce consistency of datasets
seed = 1;
err = 0.01; 
rng(seed, 'twister');

%% Generate Data
n = 4096;
dim = 4096;

[x, y, ~] = generate_dataset(n, dim, err, seed); 

alpha = 2;
epoch = 64;
num_thread = 8;
s = 0.1;

num_agent = 256;
blocksize = 128;
disp('generate data done')
%% Begin experiments
%% finito sync parallel
% expId = size(records,1) + 1;
% 
% 
% records(expId, :) = {'Finito', 'sync', 'parallel', n, dim, alpha, s, epoch, num_thread, [], zeros(1, epoch), zeros(1, epoch)};
%     
% cd('Finito/sync_parallel');
% mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" Finito_sync_parallel_mex.cpp
% 
% for tryout = 1:epoch
%     [db_trained, calculation_time] = Finito_sync_parallel_mex(x, y, alpha, s, tryout, num_thread);
%     records{expId, 11}(tryout) = calculation_time;
%     records{expId, 12}(tryout) = f(db_trained', x, y, s);
% end
% cd(cpath);

%% finito async parallel
% expId = size(records,1) + 1;
% 
% 
% records(expId, :) = {'Finito', 'async', 'parallel', n, dim, alpha, s, epoch, num_thread, [], zeros(1, epoch), zeros(1, epoch)};
%     
% cd('Finito/async_parallel');
% mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" Finito_async_parallel_mex.cpp
% 
% for tryout = 1:epoch
%     [db_trained, calculation_time] = Finito_async_parallel_mex(x, y, alpha, s, tryout, num_thread);
%     records{expId, 11}(tryout) = calculation_time;
%     records{expId, 12}(tryout) = f(db_trained', x, y, s);
% end
% cd(cpath);
%% finito async cuda
x_a = x';
x_a = x_a(:);
expId = size(records,1) + 1;

records(expId, :) = {'Finito', 'async', 'cuda', n, dim, alpha, s, epoch, num_agent, blocksize, zeros(1, epoch), zeros(1, epoch)};
    
cd('Finito/async_cuda');
mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  Finito_async_cuda_mex.cu
for tryout = 1:epoch
    [db_trained, calculation_time] = Finito_async_cuda_mex(x_a, y, alpha, s, tryout, num_agent, blocksize);
    records{expId, 11}(tryout) = calculation_time;
    records{expId, 12}(tryout) = f(db_trained', x, y, s);
end
cd(cpath);

%% finito sync cuda
x_a = x';   
x_a = x_a(:);
expId = size(records,1) + 1;

records(expId, :) = {'Finito', 'sync', 'cuda', n, dim, alpha, s, epoch, num_agent, blocksize, zeros(1, epoch), zeros(1, epoch)};
    
cd('Finito/sync_cuda');
mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  Finito_sync_cuda_mex.cu
for tryout = 1:epoch
    [db_trained, ~, calculation_time] = Finito_sync_cuda_mex(x_a, y, alpha, s, tryout, num_agent, blocksize, zeros(1, dim * n), zeros(1, dim));
    records{expId, 11}(tryout) = calculation_time;
    records{expId, 12}(tryout) = f(db_trained', x, y, s);
end
cd(cpath);



%% scd sync parallel
% expId = size(records,1) + 1;
% 
% 
% records(expId, :) = {'SCD', 'sync', 'parallel', n, dim, alpha, s, epoch, num_thread, [], zeros(1, epoch), zeros(1, epoch)};
%     
% cd('SCD/sync_parallel');
% mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" scd_sync_parallel.cpp
% 
% for tryout = 1:epoch
%     [db_trained, calculation_time] = scd_sync_parallel(x, y, alpha, s, dim * tryout, num_thread);
%     records{expId, 11}(tryout) = calculation_time;
%     records{expId, 12}(tryout) = f(db_trained', x, y, s);
% end
% cd(cpath);
%% scd async parallel
% expId = size(records,1) + 1;
% 
% 
% records(expId, :) = {'SCD', 'async', 'parallel', n, dim, alpha, s, epoch, num_thread, [], zeros(1, epoch), zeros(1, epoch)};
%     
% cd('SCD/async_parallel');
% mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" scd_async_parallel.cpp
% 
% for tryout = 1:epoch
%     [db_trained, calculation_time] = scd_async_parallel(x, y, alpha, s, dim * tryout, num_thread);
%     records{expId, 11}(tryout) = calculation_time;
%     records{expId, 12}(tryout) = f(db_trained', x, y, s);
% end
% cd(cpath);
%% scd sync cuda   
% x_a = x(:);
% expId = size(records,1) + 1;
% 
% records(expId, :) = {'SCD', 'sync', 'cuda', n, dim, alpha, s, epoch, num_agent, blocksize, zeros(1, epoch), zeros(1, epoch)};
%     
% cd('SCD/sync_cuda');
% mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  scd_sync_cuda_mex.cu
% for tryout = 1:epoch
%     [db_trained, calculation_time] = scd_sync_cuda_mex(x_a, y, alpha, s, dim * tryout, num_agent, blocksize, zeros(1, dim));
%     records{expId, 11}(tryout) = calculation_time;
%     records{expId, 12}(tryout) = f(db_trained', x, y, s);
% end
% cd(cpath);
%% scd async cuda
% x_a = x(:);
% expId = size(records,1) + 1;
% 
% records(expId, :) = {'SCD', 'async', 'cuda', n, dim, alpha, s, epoch, num_agent, blocksize, zeros(1, epoch), zeros(1, epoch)};
%     
% cd('SCD/async_cuda');
% mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  scd_async_cuda_mex.cu
% for tryout = 1:epoch
%     [db_trained, calculation_time] = scd_async_cuda_mex(x_a, y, alpha, s, dim * tryout, num_agent, blocksize, zeros(1, dim));
%     records{expId, 11}(tryout) = calculation_time;
%     records{expId, 12}(tryout) = f(db_trained', x, y, s);
% end
% cd(cpath);
%% Save results
save('records.mat', 'records')