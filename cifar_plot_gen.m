0.637175917856499

figure
for i = 2:401
    if abs(records{i,12}(32) - 0.637175917856490) > 1e-6
        continue
    end
    plot(records{i,11},records{i,12}, 'DisplayName', ...
    sprintf("%s, #processor: %d, stepsize: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}, records{i,6}));
    hold on
end
legend
xlabel("time(s)")
ylabel("function value")
title(sprintf("n=%d, dim=%d",records{i,4},records{i,5}));
% saveas(gcf, sprintf("./plots/n=%d, dim=%d",records{i,4},records{i,5}))
% saveas(gcf, ...
%     sprintf("plots/n=%d, dim=%d.jpeg",records{i,4},records{i,5}), 'jpeg')
% close(gcf)