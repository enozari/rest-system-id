function macro_lin_Izhikevic_iterator(section, sweep, varargin)
%MACRO_LIN_IZHIKEVIC_ITERATOR Code for iterating over macro_lin_Izhikevic
% and obtaining and plotting aggregate statistics of its outputs.
% 
%   Input arguments:
%  
%   section: can take four values of 'submit', 'run', 'gather', and 'plot',
%   with the same meanings as in main_fmri.m or main_ieeg.m.
% 
%   sweep: same as sweep input argument in macro_lin_Izhikevic.m.
% 
%   varargin: repetition index for section == 'run'
% 
%   Copyright (C) 2021, Erfan Nozari
%   All rights reserved.

run_on_cluster = 1;
n_rep = 100;
SNR = inf;

switch section
    case 'submit'                                                           % Section 1: submitting SGE jobs, one per subject per cross validation fold.
        if ~exist('mlIi_data', 'dir')
            mkdir mlIi_data                                                 % Directory to store the created data files. main(2) will later look inside this folder.
        end
        if run_on_cluster
            write_mlIi_sh();                                                    % Function to write the shell code main.sh for submitting SGE jobs. It can be commented out if the job is run locally.                                     
            system('chmod +x macro_lin_Izhikevic_iterator.sh');
            system('rm -f macro_lin_Izhikevic_iterator.sh.*');
            for i_rep = 1:n_rep
                if ~exist(['mlIi_data/rep_' num2str(i_rep) '.mat'], 'file')
                    system('rm -f /cbica/home/nozarie/.matlab/R2018a/toolbox_cache-9.4.0-3284594471-glnxa64.xml');
                    system(['qsub -l s_vmem=64G -l h_vmem=64G ./macro_lin_Izhikevic_iterator.sh ' ...
                        sweep ' ' num2str(i_rep)]);
                end
            end
        else
            for i_rep = 1:n_rep
                macro_lin_Izhikevic('run', sweep, i_rep)
            end
        end
    case 'run'                                                                % Section 1.5: the "inner" script running all methods for each subject-cross validation fold. This is only to be called internally by main(1), not by the user.
        i_rep = varargin{1};
        run_algorithm = 1;
        compute_R2 = 1;
        plot_graphics = 0;
        mlI_out = macro_lin_Izhikevic(sweep, SNR, run_algorithm, compute_R2, plot_graphics);

        save(['mlIi_data/rep_' num2str(i_rep) '.mat'], 'mlI_out');                               % Saving the data to be used in the subsequent main(2) call.
    case 'gather'                                                                  % Section 2: collecting the results of all jobs from main(1). Must be run after completion of all jobs on the cluster/locally using main(1). Any jobs not completed will be replaced with NaNs.
        mlI_out_rec = cell(n_rep, 1);                                        % Cell array for collecting all R^2 vectors from methods.
        
        for i_rep = 1:n_rep
            filename = ['mlIi_data/rep_' num2str(i_rep) '.mat'];  % The filename that should have been created in call to main(1).
            if exist(filename, 'file')
                load(filename, 'mlI_out')
                mlI_out_rec{i_rep} = mlI_out;
            end
        end
        mlI_out_rec = mlI_out_rec(~cellfun(@isempty, mlI_out_rec));
        mlI_out_rec = cell2mat(mlI_out_rec);                                          % Transforming from cell array to numerical array that takes less memory. Same below. This is n x n_method x n_subj.
        
        save mlIi_data.mat mlI_out_rec
    case 'plot'                                                                  % Section 3: Plotting the results
        load mlIi_data.mat mlI_out_rec
        R2_lin_rec = cell2mat({mlI_out_rec.R2_lin_rec}');
        R2_nonlin_rec = cell2mat(permute({mlI_out_rec.R2_nonlin_rec}, [1 3 2]));
        R2_nonlin_rec_max = permute(max(R2_nonlin_rec, [], 1), [3 2 1]);
        switch sweep
            case 'nave'
                par_vals = mlI_out_rec(1).nave_vals;
                x_label = 'N_{\rm ave}';
                x_dir = 'normal';
                legend_loc = 'northeast';
                ylims = [-0.5 1];
            case 'fpass'
                par_vals = mlI_out_rec(1).fpass_vals;
                x_label = 'f_{\rm cutoff}';
                x_dir = 'reverse';
                legend_loc = 'southeast';
                ylims = [0 1];
        end
        n_par = numel(par_vals);
        h_vals = mlI_out_rec(1).h_vals;
        
        hf = figure;
        hf.Position = [441   231   728   568];
        boxplot(R2_lin_rec, 'PlotStyle', 'compact', 'Whisker', inf, 'Colors', matlab_green, ...
            'Positions', par_vals)
        hold on
        boxplot(R2_nonlin_rec_max, 'PlotStyle', 'compact', 'Whisker', inf, 'Colors', matlab_yellow, ...
            'Positions', par_vals)
        ha = gca;
        boxplot_magnify(ha.Children, 4)
        set(gca, 'xtick', par_vals, ...
            'xticklabel', arrayfun(@num2str, par_vals, 'UniformOutput', 0), ...
            'xscale', 'log', 'xdir', x_dir)
        hx = xlabel(['$' x_label '$'], 'Interpreter', 'latex');
        ylabel('$R^2$', 'Interpreter', 'latex')
        ha.FontSize = 35; 
        ha.TickLabelInterpreter = 'latex';
        ha.Position(4) = ha.Position(4) - 0.1;
        ha.Position(2) = ha.Position(2) + 0.1;
        xlim([par_vals(1)-0.5 par_vals(end)+0.5])
        ylim([0 1])
        hx.Position(2) = hx.Position(2) - 30;
        hb1 = bar(-1, 0, 'FaceColor', matlab_green);
        hb2 = bar(-1, 0, 'FaceColor', matlab_yellow);
        legend([hb1 hb2], {'Linear', 'Nonlinear'}, 'fontsize', 35, 'interpreter', 'latex', ...
            'Location', legend_loc)
        grid on
        exportgraphics(gca, ['mlIi_' sweep '_boxplot.eps'])
        
        hf = figure;
        hf.Position = [26         231        1183         568];
        R2_nonlin_means = median(R2_nonlin_rec, 3);
        if isequal(sweep, 'nave')
            R2_nonlin_means(1:7, 2) = nan;
            R2_nonlin_means(1:2, 3) = nan;
            R2_nonlin_means(1, 4) = nan;
        end
        R2_nonlin_stds = iqr(R2_nonlin_rec, 3);
        errorbar(repmat(h_vals', 1, n_par), R2_nonlin_means, R2_nonlin_stds/sqrt(n_rep), '.-', ...
            'linewidth', 4, 'CapSize', 0, 'MarkerSize', 35)
        xlabel('$h$', 'Interpreter', 'latex')
        ylabel('$R^2_{\rm NL}$', 'Interpreter', 'latex')
        ha = gca;
        set(ha, 'xscale', 'log', 'fontsize', 35, 'ticklabelinterpreter', 'latex')
        legends = arrayfun(@(nave)['$' x_label ' = ' num2str(nave) '$'], par_vals, 'UniformOutput', 0);
        hl = legend(ha, legends);
        set(hl, 'fontsize', 35, 'interpreter', 'latex', 'location', 'northeastout')
        ylim(ylims)
        colors = color_sweep(matlab_blue, 2, 2, [], 0.3);
        for i_nave = 1:n_par
            ha.Children(i_nave).Color = colors(i_nave, :);
        end
        grid on
        exportgraphics(gca, ['mlIi_' sweep '_errorbar.eps'])
end
end

%% Auxiliary functions
function write_mlIi_sh                                                      % This function writes the shell script main.sh to the same directory. This is used for running jobs on a cluster. The main.sh could be directly written, but in this way all the code is accessible from MATLAB. Note that this is unnecessary if the jobs are going to be run locally.
s = ["#!/bin/bash";
    "matlab -r ""macro_lin_Izhikevic_iterator('run', '$1', $2); exit"""];
fileID = fopen('macro_lin_Izhikevic_iterator.sh', 'w');
fprintf(fileID, '%s\n', s);
end