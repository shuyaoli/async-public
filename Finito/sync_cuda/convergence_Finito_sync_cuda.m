mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_75,code=\"sm_75,compute_75\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  Finito_sync_cuda_mex.cu
% mexcuda NVCCFLAGS='-m64 -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\"'  LINKLIBS='\$LINKLIBS -L/usr/local/cuda/lib64 -lcurand'  Finito_sync_cuda_mex.cu

clear;

addpath('../../routine_work/');
Initialize_data;

num_pts = 40;
epochs = [1:num_pts];
times = zeros(1, num_pts);
function_values = zeros(1, num_pts);

x_a = x';   
x_a = x_a(:);

NUM_AGENT = 512;
BLOCKSIZE = 128;

for tryout = 1:num_pts
    [db_trained, ~, calculation_time] = ...
        Finito_sync_cuda_mex(x_a, y, alpha, s, tryout, NUM_AGENT, BLOCKSIZE, ...
        zeros(1, dim * n), zeros(1, dim));
    times(tryout) = calculation_time;
    function_values(tryout) = f(db_trained', x, y, s);
end

save(sprintf("sync_cuda,n=%d,dim=%d,s=%.1f,NUM_AGENT=%d,finito.mat",n,dim,s,NUM_AGENT), 'times', 'function_values');

ref = figure;
plot(1:num_pts, times);
xlabel("epoch")
ylabel("time (s)")
title(sprintf("time-epoch, n=%d, dim=%d, s=%.1f, #AGENT=%d",n,dim,s,NUM_AGENT));
saveas(ref,sprintf("time-epoch,sync_cuda,n=%d,dim=%d,s=%.1f,NUMAGENT=%d,finito.jpeg",n,dim,s,NUM_AGENT),'jpeg');
close(ref);

fvEpoch = figure;
plot(1:num_pts, function_values);
xlabel("epoch")
ylabel("function value")
title(sprintf("fv-epoch, n=%d, dim=%d, s=%.1f, #AGENT=%d",n,dim,s,NUM_AGENT));
saveas(fvEpoch,sprintf("fv-epoch,sync_cuda,n=%d,dim=%d,s=%.1f,NUM_AGENT=%d,finito.jpeg",n,dim,s,NUM_AGENT),'jpeg');  
close(fvEpoch);

fvTime = figure;
plot(times, function_values);
xlabel("time")
ylabel("function value")
title(sprintf("fv-time, n=%d, dim=%d, s=%.1f, #AGENT=%d",n,dim,s,NUM_AGENT));
saveas(fvTime,sprintf("fv-time,sync_cuda,n=%d,dim=%d,s=%.1f,NUM_AGENT=%d,finito.jpeg",n,dim,s,NUM_AGENT),'jpeg');  
close(fvTime);