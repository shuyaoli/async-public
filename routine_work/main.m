addpath('../routine_work/');

Initialize;

mean_z = zeros(1,dim);
z = zeros(n, dim);

sigmoid = @(x) 1./(1 + exp(-x));

for k = 1:1
    for ik = 1:n
        dotsum = x(ik,:) * mean_z';
        z(ik,:) = mean_z - alpha * (-sigmoid(- y(ik) * dotsum) * y(ik) * x(ik,:) + s * mean_z);
        
    end
    mean_z = mean(z,1);

    display(mean_z);

end