clear;load cifar.mat
[idxs, shortestTimes] = return_index(records,1e-12, 0.657181448037133, 10000,3072);

for ctr = 1:8
    i = idxs(ctr);
    fprintf("Ctr %d: %s %s %s\n", ctr, records{i, 1}, records{i,2}, records{i, 3});
end

fprintf("Finito parallel: async gives %.2f speedup\n", shortestTimes(1) / shortestTimes(2));
fprintf("Finito cuda    : async gives %.2f speedup\n", shortestTimes(5) / shortestTimes(6));
fprintf("Finito sync    : cuda  gives %.2f speedup\n", shortestTimes(1) / shortestTimes(5));
fprintf("Finito async   : cuda  gives %.2f speedup\n", shortestTimes(2) / shortestTimes(6));
fprintf('\n')

fprintf("SCD parallel: async gives %.2f speedup\n", shortestTimes(3) / shortestTimes(4));
fprintf("SCD cuda    : async gives %.2f speedup\n", shortestTimes(7) / shortestTimes(8));
fprintf("SCD sync    : cuda  gives %.2f speedup\n", shortestTimes(3) / shortestTimes(7));
fprintf("SCD async   : cuda  gives %.2f speedup\n", shortestTimes(4) / shortestTimes(8));


% figure
% for i = idxs(5:8)
%     if i > 0 
%         plot(records{i,11},records{i,12}, 'DisplayName', ...
%         sprintf("%s, #agent: %d, stepsize: %.2f", ...
%             [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}, records{i,6}));
%         hold on
%     end
% end
% legend
% xlabel("time(s)")
% ylabel("function value")
% title(sprintf("CIFAR-10, n=%d, dim=%d",10000,1024*3));
% fig  = gcf;
% fig.Units = 'inches';
% fig.Position = [0 0 10 6];

% 657181448037133 CIFAR

