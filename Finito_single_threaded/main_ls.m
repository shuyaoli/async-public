Initialize_ls;


p = zeros(n, dim); 

for i = 1:n
    p(i, :) = grad_ls(phi(i,:)', x(i,:), y(i)) - alpha * s * phi(i,:)';
end


%% Uniformly random sampling
% tic
% for k = 1: n * epoch
%         w = -1/(alpha * s * n) * sum(p, 1);
%         j = randi(n);
%         phi(j,:) = w;
%         p(j,:) = grad_fi(phi(j,:), x(j,:), y(j,:),s) - alpha * s * phi(i,:)';
%         if rem(k,500) == 0
%             fprintf('Loss: %f\n',ls(mean(phi,1)',x,y));
%         end
% end
% toc
%% Uniform random shuffle without replacement
tic
for k = 1: epoch

    sequence = randperm(n);
    for seq = sequence

        
        w = -1/(alpha * s * n) * sum(p, 1);
        
        
        phi(seq,:) = w;
        p(seq,:) = grad_ls(phi(seq,:)', x(seq,:), y(seq))- alpha * s * phi(i,:)';

    end
        if rem(epoch,100) == 0
            fprintf('Loss: %f\n',ls(mean(phi,1)',x,y));
        end
end
toc
%% Retrieve result


% final_sum = 0;
% for i = 1:n
%     final_sum = final_sum + fi(db_trained',x(i,:),y(i),s);
% end
% assert(final_sum/n == f(db_trained',x,y,s))