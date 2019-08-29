function idx = return_index_helper(records, cut, algorithmQ, syncQ, parallelQ)

    idx = -1;
    shortestTime = Inf;
    for i = 2:201
        if strcmp(records{i,1}, algorithmQ) && strcmp(records{i,2}, syncQ) && strcmp(records{i,3}, parallelQ)
            thisShortestTime= min(records{i,11} (abs(records{i,12} - 0.657181448037133) < cut));
            if (thisShortestTime < shortestTime)
                shortestTime = thisShortestTime;
                idx = i;
            end
        end
    end

end