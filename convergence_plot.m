clear;
load('records.server.mat') % algorithm, n, dim, alpha, s, epoch, p, blocksize, times, function values

figure;
% for i = 2:5
%     plot(records{i,11},records{i,12}, 'DisplayName', ...
%     sprintf([records{i,1}, '  ', records{i,2}, '  ',records{i,3}]));
%     hold on
% end

% for i = 10:13
%     plot(records{i,11},records{i,12}, 'DisplayName', ...
%     sprintf([records{i,1}, '  ', records{i,2}, '  ',records{i,3}]));
%     hold on
% end

% local
% finito sync cuda: 256, 1024, 2048, very closed;  256 fastest
% finito async cuda: 1024 512 256 closed; 256 fastest
% scd sync cuda: 4096 fastest
% scd async cuda: 512 256 1024 closed; 256 fastest
% for i = 12:4:33 % change starting point
%     plot(records{i,11},records{i,12}, 'DisplayName', ...
%     sprintf("%s, $processor: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}));
%     hold on
% end

% server

for i = 6:2:15
    plot(1:64,records{i,11}, 'DisplayName', ...
    sprintf("%s, $processor: %d", [records{i,1}, '  ', records{i,2}, '  ',records{i,3}], records{i,9}));
    hold on
end

legend
