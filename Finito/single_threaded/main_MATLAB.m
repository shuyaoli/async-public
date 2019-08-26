addpath('../../routine_work/');

Initialize;
rng('shuffle','twister');
phi = zeros(n,dim);

p = zeros(n, dim); 

for i = 1:n
    p(i, :) = grad_fi(phi(i,:), x(i,:), y(i),s)- alpha * s * phi(i,:);
end


%% Uniformly random sampling
tic
for k = 1: n * epoch
    
        w = -1/(alpha * s * n ) * sum(p, 1);
        j = randi(n);
        phi(j,:) = w;
        p(j,:) = grad_fi(phi(j,:), x(j,:), y(j),s) - alpha * s * phi(j,:);
        
        
        if rem(k, 3000) == 1

            
            db_trained = mean(phi, 1); 

            result = x * db_trained' > 0;
            result = 2 * result - 1;

            error_rate = 1 - sum(result == y) / size(result,1);

        
            fprintf('Itr: %d. Error: %f. Cost function: %.10f.\n', k, error_rate, (f(mean(phi, 1)', x, y, s)));
        end
end
toc
%% No longer updated and maintaned - Uniform random shuffle without replacement
% tic
% for k = 1: epoch
% 
%     sequence = randperm(n);
%     for seq = sequence
% 
%         
%         w = -1/(alpha * s * n) * sum(p, 1);
%         
%         
%         phi(seq,:) = w;
%         p(seq,:) = grad_fi(phi(seq,:), x(seq,:), y(seq),s) - alpha * s * phi(i,:);
%         
%         if rem(seq, 1000) == 1
%             db_trained = mean(phi, 1); 
% 
%             result = x * db_trained' > 0;
%             result = 2 * result - 1;
% 
%             error_rate = 1 - sum(result == y) / size(result,1);
% 
%         
%             fprintf("Epoch: %d. Error: %f. Cost function: %.10f.\n", k, error_rate, (f(mean(phi, 1)', x, y, s)));
%         end
%     end
% 
% end
% toc
%% Retrieve result
db_trained = mean(phi, 1) 

result = x * db_trained' > 0;
result = 2 * result - 1;

error_rate = 1 - sum(result == y) / size(result,1)
fprintf('%.10f\n',f(mean(phi, 1)', x, y, s));
% final_sum = 0;
% for i = 1:n
%     final_sum = final_sum + fi(db_trained',x(i,:),y(i),s);
% end
% assert(final_sum/n == f(db_trained',x,y,s))

rmpath('../routine_work/');