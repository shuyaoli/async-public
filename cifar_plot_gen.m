clear;load cifar.mat

[idxs] = return_index(records,1e-8);
        
figure
for i = idxs(5:8)
    if i > 0 
        plot(records{i,11},records{i,12}, 'DisplayName', ...
        sprintf("%s, #agent: %d, stepsize: %.2f", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}, records{i,6}));
        hold on
    end
end
legend
xlabel("time(s)")
ylabel("function value")
title(sprintf("CIFAR-10, n=%d, dim=%d",10000,1024*3));
% saveas(gcf, sprintf("./plots/n=%d, dim=%d",records{i,4},records{i,5}))
% saveas(gcf, ...
%     sprintf("plots/n=%d, dim=%d.jpeg",records{i,4},records{i,5}), 'jpeg')
% close(gcf)