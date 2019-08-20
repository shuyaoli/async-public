function records = scd_parallel_generate(records, x, y, n, dim, alpha, s, epoch, num_thread, cpath)
% scd sync parallel
expId = size(records,1) + 1;

records(expId, :) = {'SCD', 'sync', 'parallel', n, dim, alpha, s, epoch, num_thread, [], zeros(1, epoch), zeros(1, epoch)};
    
cd('SCD/sync_parallel');
mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" scd_sync_parallel.cpp

for tryout = 1:epoch
    [db_trained, calculation_time] = scd_sync_parallel(x, y, alpha, s, dim * tryout, num_thread);
    records{expId, 11}(tryout) = calculation_time;
    records{expId, 12}(tryout) = f(db_trained', x, y, s);
end
cd(cpath);
% scd async parallel
expId = size(records,1) + 1;


records(expId, :) = {'SCD', 'async', 'parallel', n, dim, alpha, s, epoch, num_thread, [], zeros(1, epoch), zeros(1, epoch)};
    
cd('SCD/async_parallel');
mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" scd_async_parallel.cpp

for tryout = 1:epoch
    [db_trained, calculation_time] = scd_async_parallel(x, y, alpha, s, dim * tryout, num_thread);
    records{expId, 11}(tryout) = calculation_time;
    records{expId, 12}(tryout) = f(db_trained', x, y, s);
end
cd(cpath);