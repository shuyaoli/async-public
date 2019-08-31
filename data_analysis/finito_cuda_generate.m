function records = finito_cuda_generate(records, x, y, n, dim, alpha, s, epoch, num_agent, blocksize, COMP, cpath)
%% finito sync cuda
x_a = x';   
x_a = x_a(:);
expId = size(records,1) + 1;

records(expId, :) = {'Finito', 'sync', 'cuda', n, dim, alpha, s, epoch, num_agent, blocksize, zeros(1, epoch), zeros(1, epoch)};
    
cd('Finito/sync_cuda');
% if (COMP == 30)
%     mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  Finito_sync_cuda_mex.cu
% elseif (COMP == 75)
%     mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_75,code=\"sm_75,compute_75\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  Finito_sync_cuda_mex.cu
% end

for tryout = 1:epoch
    [db_trained, ~, calculation_time] = Finito_sync_cuda_mex(x_a, y, alpha, s, tryout, num_agent, blocksize, zeros(1, dim * n), zeros(1, dim));
    records{expId, 11}(tryout) = calculation_time;
    records{expId, 12}(tryout) = f(db_trained', x, y, s);
end
cd(cpath);

%% finito async cuda
x_a = x';
x_a = x_a(:);
expId = size(records,1) + 1;

records(expId, :) = {'Finito', 'async', 'cuda', n, dim, alpha, s, epoch, num_agent, blocksize, zeros(1, epoch), zeros(1, epoch)};
    
cd('Finito/async_cuda');
% if COMP == 30
%     mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  Finito_async_cuda_mex.cu
% elseif COMP == 75
%     mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_75,code=\"sm_75,compute_75\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  Finito_async_cuda_mex.cu
% end

for tryout = 1:epoch
    [db_trained, calculation_time] = Finito_async_cuda_mex(x_a, y, alpha, s, tryout, num_agent, blocksize);
    records{expId, 11}(tryout) = calculation_time;
    records{expId, 12}(tryout) = f(db_trained', x, y, s);
end
cd(cpath);
