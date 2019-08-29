function records = scd_cuda_generate (records, x, y, n, dim, alpha, s, epoch, num_agent, blocksize, COMP, cpath)
%% scd sync cuda   
x_a = x(:);
expId = size(records,1) + 1;

records(expId, :) = {'SCD', 'sync', 'cuda', n, dim, alpha, s, epoch, num_agent, blocksize, zeros(1, epoch), zeros(1, epoch)};
    
cd('SCD/sync_cuda');
% if COMP == 30
%     mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  scd_sync_cuda_mex.cu
% elseif COMP == 75
%     mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_75,code=\"sm_75,compute_75\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  scd_sync_cuda_mex.cu
% end

for tryout = 1:epoch
    [db_trained, calculation_time] = scd_sync_cuda_mex(x_a, y, alpha, s, dim * tryout, num_agent, blocksize, zeros(1, dim));
    records{expId, 11}(tryout) = calculation_time;
    records{expId, 12}(tryout) = f(db_trained', x, y, s);
end
cd(cpath);

%% scd async cuda
x_a = x(:);
expId = size(records,1) + 1;

records(expId, :) = {'SCD', 'async', 'cuda', n, dim, alpha, s, epoch, num_agent, blocksize, zeros(1, epoch), zeros(1, epoch)};
    
cd('SCD/async_cuda');
% if COMP == 30
%     mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  scd_async_cuda_mex.cu
% elseif COMP == 75
%     mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_75,code=\"sm_75,compute_75\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  scd_async_cuda_mex.cu
% end

for tryout = 1:epoch
    [db_trained, calculation_time] = scd_async_cuda_mex(x_a, y, alpha, s, dim * tryout, num_agent, blocksize, zeros(1, dim));
    records{expId, 11}(tryout) = calculation_time;
    records{expId, 12}(tryout) = f(db_trained', x, y, s);
end
cd(cpath);