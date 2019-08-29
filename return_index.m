function idxs= return_index(records,cut)

idxs = [];

for parallelQ = {'parallel' 'cuda'}     
    for algorithmQ = {'Finito' 'SCD'}
        for syncQ = {'sync' 'async'}
            idx = return_index_helper(records, cut, algorithmQ, syncQ, parallelQ);
            idxs = [idxs idx];
        end
    end
end
            
end