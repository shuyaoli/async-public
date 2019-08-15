clear;
load('records.mat') % algorithm, n, dim, alpha, s, epoch, p, blocksize, times, function values

figure;
for i = [31 35 43]
    plot(records{i,11},records{i,12}, 'DisplayName', sprintf(records{i,1})) ;hold on
end

legend
