clear;

addpath('../../routine_work/');
Initialize_data;

num_pts = 40;
epochs = [1:num_pts];
times = zeros(1, num_pts);
function_values = zeros(1, num_pts);

numthread = 16;

for tryout = 1:num_pts
    
    [db_trained, calculation_time] = Finito_multi_threaded(x, y, alpha, s, epoch, numthread);
    times(tryout) = calculation_time;
    function_values(tryout) = f(db_trained', x, y, s);

end
