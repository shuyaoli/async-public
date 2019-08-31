function [idx, shortestTime] = return_index_helper(records, cut, algorithmQ, syncQ, parallelQ, machineMin, n, dim)

    idx = -1;
    shortestTime = Inf;
    for i = 2:size(records,1)
        if strcmp(records{i,1}, algorithmQ) && strcmp(records{i,2}, syncQ) && strcmp(records{i,3}, parallelQ) ...
                && records{i, 4} == n && records{i, 5} == dim
            thisShortestTime= min(records{i,11} (abs(records{i,12} - machineMin) < cut));
            if (thisShortestTime < shortestTime)
                shortestTime = thisShortestTime;
                idx = i;
            end
        end
    end

end

