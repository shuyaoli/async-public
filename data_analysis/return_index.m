function [idxs, shortestTimes]= return_index(records,cut, machineMin, n, dim)

idxs = [];
shortestTimes = [];
for parallelQ = {'parallel' 'cuda'}     
    for algorithmQ = {'Finito' 'SCD'}
        for syncQ = {'sync' 'async'}
            [idx, shortestTime]= return_index_helper(records, cut, algorithmQ, syncQ, parallelQ, machineMin, n, dim);
            idxs = [idxs idx];
            shortestTimes = [shortestTimes, shortestTime];
        end
    end
end
            
end