mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" Finito_multi_threaded.cpp
clear;

addpath('../../routine_work/');
Initialize_data;

num_pts = 40;
epochs = [1:num_pts];
times = zeros(1, num_pts);
function_values = zeros(1, num_pts);

numthread = 8;

for tryout = 1:num_pts
    
    [db_trained, calculation_time] = Finito_multi_threaded(x, y, alpha, s, tryout, numthread);
    times(tryout) = calculation_time;
    function_values(tryout) = f(db_trained', x, y, s);

end

save(sprintf("sync_parallel,n=%d,dim=%d,s=%.1f,finito.mat",n,dim,s), 'times', 'function_values');

ref = figure;
plot(1:num_pts, times);
xlabel("epoch")
ylabel("time (s)")
title(sprintf("time-epoch, n=%d, dim=%d, s=%.1f",n,dim,s));
saveas(ref,sprintf("time-epoch,sync_parallel,n=%d,dim=%d,s=%.1f,finito.jpeg",n,dim,s),'jpeg');
close(ref);

fvEpoch = figure;
plot(1:num_pts, function_values);
xlabel("epoch")
ylabel("function value")
title(sprintf("fv-epoch, n=%d, dim=%d, s=%.1f",n,dim,s));
saveas(fvEpoch,sprintf("fv-epoch,sync_parallel,n=%d,dim=%d,s=%.1f,finito.jpeg",n,dim,s),'jpeg');  
close(fvEpoch);

fvTime = figure;
plot(times, function_values);
xlabel("time")
ylabel("function value")
title(sprintf("fv-time, n=%d, dim=%d, s=%.1f",n,dim,s));
saveas(fvTime,sprintf("fv-time,sync_parallel,n=%d,dim=%d,s=%.1f,finito.jpeg",n,dim,s),'jpeg');  
close(fvTime);
