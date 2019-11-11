clear;
load('records.server.mat') % algorithm, n, dim, alpha, s, epoch, p, blocksize, times, function values
%% server
% Cutoff value to exclude outliers
% 0.432648679303562 8192  8192
% 0.531309702952792 16384 2048
% 0.195167411551043 2048  16384
para_n = 2048;
para_d = 16384;
cutoff = 0.195167411551043;
[idxs] = return_index(records,1e-6, cutoff, para_n, para_d);   
h = figure;

for i = idxs(5:6)
    if i > 0 
        plot(records{i,11},(records{i,12} - cutoff),'DisplayName', ...
        sprintf("%6s %5s %s", records{i,1}, records{i,2},"GPU"),...
        'LineWidth', 3);
        hold on
    end
end
for i = idxs(7:8)
    if i > 0 
        plot(records{i,11},(records{i,12} - cutoff),'--', 'DisplayName', ...
        sprintf("%5.25s %6s %s", records{i,1}, records{i,2},'GPU'),...
        'LineWidth', 3);
        hold on
    end
end
legend('fontsize', 10);
xlabel("time(s)", 'fontsize',16)
ylabel("function value suboptimality", 'fontsize',16)
title(sprintf("plot with normal scale, n=%d, d=%d",para_n, para_d),'fontsize',18);
fig  = gcf;
fig.Units = 'inches';
fig.Position = [0 0 10 6];


% [idxs, shortestTimes] = return_index(records,1e-6, 0.195167411551043, 2048,16384);
% for ctr = 1:8
%     i = idxs(ctr);
%     fprintf("Ctr %d: %s %s %s\n", ctr, records{i, 1}, records{i,2}, records{i, 3});
% end

% fprintf("Finito parallel: async gives %.2f speedup\n", shortestTimes(1) / shortestTimes(2));
% fprintf("Finito cuda    : async gives %.2f speedup\n", shortestTimes(5) / shortestTimes(6));
% fprintf("Finito sync    : cuda  gives %.2f speedup\n", shortestTimes(1) / shortestTimes(5));
% fprintf("Finito async   : cuda  gives %.2f speedup\n", shortestTimes(2) / shortestTimes(6));
% fprintf('\n')
% fprintf("SCD parallel: async gives %.2f speedup\n", shortestTimes(3) / shortestTimes(4));
% fprintf("SCD cuda    : async gives %.2f speedup\n", shortestTimes(7) / shortestTimes(8));
% fprintf("SCD sync    : cuda  gives %.2f speedup\n", shortestTimes(3) / shortestTimes(7));
% fprintf("SCD async   : cuda  gives %.2f speedup\n", shortestTimes(4) / shortestTimes(8));


%%
% for ctr = 2:4:261
%     figure
%     for i = ctr:(ctr+3)
%         plot(1:32,records{i,12}, 'DisplayName', ...
%         sprintf([records{i,1}, '  ', records{i,2}, '  ',records{i,3}]));
%         hold on
%     end
%     xlabel('epoch')
%     ylabel('function value')
%     title(sprintf("n=%d, dim=%d, alpha=%.1f, #processor=%d",records{i,4},records{i,5},records{i,6},records{i,9}))
%     legend
%     saveas(gcf, ...
%         sprintf("plots/algorithm_comparison/fv-epoch, n=%d, dim=%d, alpha=%.1f, #processor=%d.jpeg",...
%         records{i,4},records{i,5},records{i,6},records{i,9}), 'jpeg')
%     close(gcf)
% end

%%
% % % % % local
% % % % % finito sync cuda: 256, 1024, 2048, very closed;  256 fastest
% % % % % finito async cuda: 1024 512 256 closed; 256 fastest
% % % % % scd sync cuda: 4096 fastest
% % % % % scd async cuda: 512 256 1024 closed; 256 fastest
% % % % % for i = 12:4:33 % change starting point
% % % % %     plot(records{i,11},records{i,12}, 'DisplayName', ...
% % % % %     sprintf("%s, $processor: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}));
% % % % %     hold on
% % % % % end

%% server
% for ctr = 2:20:261
%     for al = 0:3
%         figure
%         for i = (ctr+al):4:(ctr+20)
%             plot(1:32,records{i,12}, 'DisplayName', ...
%             sprintf("%s, #processor: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}));
%             hold on
%         end
%         legend
%         xlabel('epoch')
%         ylabel('function value')
%         title(sprintf("n=%d, dim=%d, alpha=%.1f",records{i,4},records{i,5},records{i,6}))
%         legend
%         saveas(gcf, ...
%             sprintf("plots/#processor_comparison/fv-epoch, n=%d, dim=%d, alpha=%.1f.jpeg",...
%             records{i,4},records{i,5},records{i,6}), 'jpeg')
%         close(gcf)
%     end
% end

%% 
% figure
% for i = 2:81
%     if records{i,12}(32) > 0.432648679308
%         continue
%     end
%     plot(records{i,11},records{i,12}, 'DisplayName', ...
%     sprintf("%s, #processor: %d, stepsize: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}, records{i,6}));
%     hold on
% end
% legend
% xlabel("time(s)")
% ylabel("function value")
% title(sprintf("n=%d, dim=%d",records{i,4},records{i,5}));
% saveas(gcf, sprintf("./plots/n=%d, dim=%d",records{i,4},records{i,5}))
% saveas(gcf, ...
%     sprintf("plots/n=%d, dim=%d.jpeg",records{i,4},records{i,5}), 'jpeg')
% close(gcf)


% figure
% for i = 262:277
%     if records{i,12}(32) > 0.43266 %0.432648679308
%         continue
%     end
% 
%     plot(records{i,11},records{i,12}, 'DisplayName', ...
%     sprintf("%s, #processor: %d, stepsize: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}, records{i,6}));
%     hold on
% end
% legend
% xlabel("time(s)")
% ylabel("function value")
% title(sprintf("n=%d, dim=%d, parallel",records{i,4},records{i,5}));
% saveas(gcf, sprintf("./plots/n=%d, dim=%d, parallel",records{i,4},records{i,5}))
% saveas(gcf, ...
%     sprintf("plots/n=%d, dim=%d, parallel.jpeg",records{i,4},records{i,5}), 'jpeg')
% close(gcf)

% cuda = inf;
% cut = 0.43264867931;
% for i = 2:81
%     
%     temp = min(records{i, 11} (records{i,12} < cut)); %432648679308
%     if (temp < cuda)
%         cuda = temp;
%     end
% end
% 
% parallel = inf;
% for i = 262:277
%     
%     temp = min(records{i, 11} (records{i,12} < cut));
%     if (temp < parallel)
%         parallel = temp;
%     end
% end
% 
% fprintf("Cuda gives %.1fx speed up\n", parallel / cuda) % 20x speed up

% figure
% for i = 82:161
%     if records{i,12}(32) > 0.5313097029527912
%         continue
%     end
%     plot(records{i,11},records{i,12}, 'DisplayName', ...
%     sprintf("%s, #processor: %d, stepsize: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}, records{i,6}));
%     hold on
% end
% legend
% xlabel("time(s)")
% ylabel("function value")
% title(sprintf("n=%d, dim=%d",records{i,4},records{i,5}));
% saveas(gcf, sprintf("./plots/n=%d, dim=%d",records{i,4},records{i,5}))
% saveas(gcf, ...
%     sprintf("plots/n=%d, dim=%d.jpeg",records{i,4},records{i,5}), 'jpeg')
% close(gcf)

% figure
% for i = 278:293
%     if records{i,12}(32) > 0.53131
%         continue
%     end
% 
%     plot(records{i,11},records{i,12}, 'DisplayName', ...
%     sprintf("%s, #processor: %d, stepsize: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}, records{i,6}));
%     hold on
% end
% legend
% xlabel("time(s)")
% ylabel("function value")
% title(sprintf("n=%d, dim=%d, parallel",records{i,4},records{i,5}));
% saveas(gcf, sprintf("./plots/n=%d, dim=%d, parallel",records{i,4},records{i,5}))
% saveas(gcf, ...
%     sprintf("plots/n=%d, dim=%d, parallel.jpeg",records{i,4},records{i,5}), 'jpeg')
% close(gcf)

% cuda = inf;
% cut = 0.53130971;
% for i = 82:161
%     
%     temp = min(records{i, 11} (records{i,12} < cut)); 
%     if (temp < cuda)
%         cuda = temp;
%     end
% end
% 
% parallel = inf;
% for i = 278:293
%     
%     temp = min(records{i, 11} (records{i,12} < cut));
%     if (temp < parallel)
%         parallel = temp;
%     end
% end
% 
% fprintf("Cuda gives %.1fx speed up\n", parallel / cuda) % 20x-30x speed up

% figure
% for i = 162:261
%     if records{i,12}(32) > 0.19516741167
%         continue
%     end
%     plot(records{i,11},records{i,12}, 'DisplayName', ...
%     sprintf("%s, #processor: %d, stepsize: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}, records{i,6}));
%     hold on
% end
% legend
% xlabel("time(s)")
% ylabel("function value")
% title(sprintf("n=%d, dim=%d",records{i,4},records{i,5}));
% saveas(gcf, sprintf("./plots/n=%d, dim=%d",records{i,4},records{i,5}))
% saveas(gcf, ...
%     sprintf("plots/n=%d, dim=%d.jpeg",records{i,4},records{i,5}), 'jpeg')
% close(gcf)

% figure
% for i = 294:309
%     if records{i,12}(32) > 0.196
%         continue
%     end
% 
%     plot(records{i,11},records{i,12}, 'DisplayName', ...
%     sprintf("%s, #processor: %d, stepsize: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}, records{i,6}));
%     hold on
% end
% legend
% xlabel("time(s)")
% ylabel("function value")
% title(sprintf("n=%d, dim=%d, parallel",records{i,4},records{i,5}));
% saveas(gcf, sprintf("./plots/n=%d, dim=%d, parallel",records{i,4},records{i,5}))
% saveas(gcf, ...
%     sprintf("plots/n=%d, dim=%d, parallel.jpeg",records{i,4},records{i,5}), 'jpeg')
% close(gcf)

% cuda = inf;
% cut = 0.19516742;
% for i = 162:261
%     
%     temp = min(records{i, 11} (records{i,12} < cut)); 
%     if (temp < cuda)
%         cuda = temp;
%     end
% end
% 
% parallel = inf;
% for i = 294:309
%     
%     temp = min(records{i, 11} (records{i,12} < cut));
%     if (temp < parallel)
%         parallel = temp;
%     end
% end
% 
% fprintf("Cuda gives %.1fx speed up\n", parallel / cuda) % 22x speed up

%% 
% sync = inf;
% async = inf;
% cut = 0.43264869; %195167411551045
% 
% for i = 2:4:81
%     temp = min(records{i, 11} (records{i,12} < cut));
%     if (temp < sync)
%         sync = temp;
%         syncid = i;
%     end
% end
% 
% for i = 3:4:81
%     
%     temp = min(records{i, 11} (records{i,12} < cut)); 
%     if (temp < async)
%         async = temp;
%         asyncid = i;
%     end
% end
% 
% 
% 
% fprintf("async gives %.1fx speed up\n", sync / async) 
