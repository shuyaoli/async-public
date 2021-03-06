clear;
SAVE_CUDA_DATA=0;
if (SAVE_CUDA_DATA)
   input("\nDo you really want to save CUDA DATA? Press Ctrl-C to break, Enter to continue");
end

Initialize_data;

if SAVE_CUDA_DATA
    if n == 4000 && dim == 300
        x_a = x';
        x_a = x_a(:);
        writematrix(x_a, '../data/SMALL/x_a');
        writematrix(y, '../data/SMALL/y')
    end
    
    if n == 4096 && dim == 256
        x_a = x';
        x_a = x_a(:);
        writematrix(x_a, '../data/MID/x_a');
        writematrix(y, '../data/MID/y')
    end
    
    if n == 8192 && dim == 1024
        x_a = x';
        x_a = x_a(:);
        writematrix(x_a, '../data/MEDIUM/x_a');
        writematrix(y, '../data/MEDIUM/y')
    end
    
    if n == 16384 && dim == 2048
        x_a = x';
        x_a = x_a(:);
        writematrix(x_a, '../data/LARGE/x_a');
        writematrix(y, '../data/LARGE/y')
    end
    
    if n == 16384 && dim == 4096
        x_a = x';
        x_a = x_a(:);
        writematrix(x_a, '../data/HUGE/x_a');
        writematrix(y, '../data/HUGE/y')
    end

    if n == 16384 && dim == 8192
        x_a = x';
        x_a = x_a(:);
        writematrix(x_a, '../data/HUGE_SERVER/x_a');
        writematrix(y, '../data/HUGE_SERVER/y')
    end
end
