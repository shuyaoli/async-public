clear;
seed = 1;
err = 0.01; 
rng(seed);

n = 4096; 
dim = 32; 

alpha = 0.5;
epoch = 20;

s = 1; % regularizer

[x, y, ~] = generate_dataset(n, dim, err, seed); 

mean_z = zeros(1,dim);
z = zeros(n, dim);

sigmoid = @(x) 1./(1 + exp(-x));

for k = 1:epoch
    for ik = 1:n
        dotsum = x(ik,:) * mean_z';
        z(ik,:) = mean_z - alpha * (-1.0 / (1+exp(y(ik) * dotsum)) * y(ik) * x(ik,:) + s * mean_z);
        
    end
    mean_z = mean(z,1);
end
fprintf('The decision boundary is %.15f, %.15f, %.15f, %.15f\n', mean_z(1),mean_z(2),mean_z(3),mean_z(4));
