function records = finito_parallel_generate(records, x, y, n, dim, alpha, s, epoch, num_thread, cpath)

%% finito sync parallel
expId = size(records,1) + 1;


records(expId, :) = {'Finito', 'sync', 'parallel', n, dim, alpha, s, epoch, num_thread, [], zeros(1, epoch), zeros(1, epoch)};
    
cd('Finito/sync_parallel');
mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" Finito_sync_parallel_mex.cpp

for tryout = 1:epoch
    [db_trained, calculation_time] = Finito_sync_parallel_mex(x, y, alpha, s, tryout, num_thread);
    records{expId, 11}(tryout) = calculation_time;
    records{expId, 12}(tryout) = f(db_trained', x, y, s);
end
cd(cpath);

%% finito async parallel
expId = size(records,1) + 1;


records(expId, :) = {'Finito', 'async', 'parallel', n, dim, alpha, s, epoch, num_thread, [], zeros(1, epoch), zeros(1, epoch)};
    
cd('Finito/async_parallel');
mex CXXFLAGS="\$CXXFLAGS -std=c++17 -O3 -latomic" Finito_async_parallel_mex.cpp

for tryout = 1:epoch
    [db_trained, calculation_time] = Finito_async_parallel_mex(x, y, alpha, s, tryout, num_thread);
    records{expId, 11}(tryout) = calculation_time;
    records{expId, 12}(tryout) = f(db_trained', x, y, s);
end
cd(cpath);
