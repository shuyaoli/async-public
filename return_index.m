function idxs= return_index(records,cut, machineMin, n, dim)

idxs = [];

for parallelQ = {'parallel' 'cuda'}     
    for algorithmQ = {'Finito' 'SCD'}
        for syncQ = {'sync' 'async'}
            idx = return_index_helper(records, cut, algorithmQ, syncQ, parallelQ, machineMin, n, dim);
            idxs = [idxs idx];
        end
    end
end
            
end