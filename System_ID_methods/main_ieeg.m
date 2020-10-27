function main_ieeg(section, varargin)
%MAIN_IEEG The wrapper code for reproducing the analyses and graphics 
% reported in the manuscript 
% E. Nozari et. al., "Is the brain macroscopically linear? A system
% identification of resting state dynamics", 2020.
%
%   main_ieeg(section), where section = 1, 1.5, 2, or 3 runs the
%   corresponding section of the script. The sections are supposed to be
%   run in order, and each one should only be run if the previous sections
%   have been completed. Only sections 1, 2, and 3 are for external call,
%   whereas section 1.5 is internally called by section 1.
% 
%   main_ieeg(section, max_jobs) with section = 1 optionally determines the
%   maximum number of jobs to be submitted to the cluster at any point in
%   time. This is ignored if run_on_cluster = 0.
% 
%   As stated above, the workflow is as follows: first run main_ieeg(1),
%   and wait until it finishes running. Since this typically takes a large
%   amount of time, it is by default assumed that the code is run on a
%   cluster with Sun Grid Engine (SGE) job scheduler. If not, set
%   run_on_cluster = 0. When compelete, run main_ieeg(2) to collect the
%   results of main_ieeg(1) and save them in main_data.mat. When complete,
%   run main_ieeg(3) to plot the graphics reported in the paper.
% 
%   Note that when the code is run on cluster with limited number of MATLAB
%   licenses, max_jobs should be set to as many MATLAB licenses as one can
%   use. If so, then main_ieeg(1, max_jobs) needs to be called over and
%   over until all the jobs get completed. The code automatically takes
%   care of skipping the jobs that have been completed before, and killing
%   the ones that have remained hung.
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

% Adding the parent directory and its sub-directories to the path
full_filename = mfilename('fullpath');
slash_loc = strfind(full_filename, '/');
addpath(genpath(full_filename(1:slash_loc(end))))

run_on_cluster = 1;                                                         % Switch to determine if this code is run on a SGE-managed cluster, where different subject data can be processed in parallel, or not, where all subjects will be run in sequence.

n_segment = 8490;                                                           % Number of total 10-second resting state iEEG recordings
switch section
    case 1                                                                  % Section 1: submitting SGE jobs (or running them locally), one per 10-second segment.
        if ~exist('main_data', 'dir')
            mkdir main_data                                                 % Directory to store the created data files. main_ieeg(2) will later look inside this folder to gather results.
        end
        if exist('nan_segments.mat', 'file')                                % nan_segments.mat is an evolving list of segments that invlude any nan entries in them so that they are ignored and not re-run indefinitely as if they were missed.
            load nan_segments.mat nan_segments
        else
            nan_segments = [];
        end
        load skip_segments.mat skip_segments                                % skip_segments.mat is an evolving list of segments that need to be skipped for reasons other than involving nan entires. The only reason encountered in our study was the ill-conditionedness of an intermediate matrix in the subspace ID method that makes that method unable to complete.
        skip_segments = union(skip_segments, nan_segments);
        
        if run_on_cluster
            if ~isempty(varargin)
                max_jobs = varargin{1};                                     % If a second optional parameter is provided in calling main_ieeg(1, max_jobs), it indicates the maximum amount of jobs that should be submitted to the cluster at once. 
            else
                max_jobs = inf;
            end
            write_main_sh();                                                % Function to write the shell code main.sh for submitting SGE jobs.
            system('chmod +x main.sh');
            
            if exist('job_id_rec.mat', 'file')
                load job_id_rec.mat job_id_rec                              % job_id_rec is a record of all the jobs that have been actively running on the cluster since the last call to this function
            else
                job_id_rec = nan(0, 2);
            end
            try
                [~, cmdout] = system('qstat');                              % Getting the latest stats of which jobs are running on the cluster
            catch
                warning('Calling qstat in main_ieeg failed, returning prematurely ...')
                return
            end
            newline_ind = strfind(cmdout, newline);                         % The following few lines of code parse the text output of qstat to extract the list of currently running jobs
            if numel(newline_ind) > 3
                cmdout = cmdout(newline_ind(3)+1:end);
                newline_ind = strfind(cmdout, newline);
                cmdout = reshape(cmdout, [], numel(newline_ind))';
                job_id_active = str2num(cmdout(:, 1:7));                    % List of currently running jobs
            else
                job_id_active = [];
            end
                
            job_id_rec_isactive = ismember(job_id_rec(:, 2), job_id_active);
            job_id_rec = job_id_rec(job_id_rec_isactive, :);                % Updating job_id_rec by removing the jobs that have completed since the last call
            
            if ~exist('trash', 'dir')                                       % Folder to save the output and error logs of completed/deleted jobs for potential further inspection
                mkdir trash
            end
            [~, cmdout] = system('find main.sh.o*');                        % The following few lines find all job output logs, sifts the ones whose corresponding job is no longer active, and moves them as well as the corresponding error logs to the trash folder. They can be inspected later if needed, or earased.
            newline_ind = strfind(cmdout, newline);
            cmdout = reshape(cmdout, newline_ind(1), [])';
            job_id_inactive = setdiff(str2num(cmdout(:, 10:16)), job_id_active);
            for i = 1:numel(job_id_inactive)
                system(['mv main.sh.*' num2str(job_id_inactive(i)) ' trash']);
            end
            
            n_jobs = 0;                                                     % Number of submitted jobs in this round, needed to ensure the number of all active jobs does not exceed max_jobs
            for i_segment = 1:n_segment
                if ~exist(['main_data/' num2str(i_segment) '.mat'], 'file') ... && 
                        && ~ismember(i_segment, job_id_rec(:, 1)) ... &&
                        && ~ismember(i_segment, skip_segments) ... &&
                        && n_jobs < max_jobs - numel(job_id_active)         % First it checks whether that segment has been completed before. This is useful if max_jobs is not infinity, so main_ieeg(1, max_jobs) needs to be called multiple times. Then it checks if that job is currently being run on a node, if it is marked to be skipped, and finally if there is room for submitting new jobs.
                    [~, cmdout] = system(['qsub -l s_vmem=64G -l h_vmem=64G ./main.sh ' num2str(i_segment)]); % Submitting the job to the cluster
                    job_id_rec(end+1, :) = [i_segment, str2double(cmdout(strfind(cmdout, 'Your job ')+(9:15)))]; % Keeping record of the just-submitted job
                    n_jobs = n_jobs + 1;
                end
            end
            
            save job_id_rec.mat job_id_rec
        else                                                                % If run_on_cluster = 0, the jobs are serially run on the current machine until complete.
            for i_segment = setdiff(1:n_segment, skip_segments)
                main_ieeg(1.5, i_segment)
            end
        end
    case 1.5                                                                % Section 1.5: the "inner" script running all methods for each segment. This is only to be called internally by main_ieeg(1), not by the user.
        i_segment = varargin{1};
        load(['rs_5min/rand_segments/Y_' num2str(i_segment) '.mat'], 'Y')
        if any(isnan(Y(:)))
            if exist('nan_segments.mat', 'file')
                load nan_segments.mat nan_segments
            else
                nan_segments = [];
            end
            nan_segments(end+1) = i_segment;
            nan_segments = unique(nan_segments);
            save nan_segments.mat nan_segments
        else
            try
                [model_summary, R2, runtime, whiteness_p] = all_methods_ieeg(Y);          % Calling the routine that runs all system id methods.
                save(['main_data/' num2str(i_segment) '.mat'], 'model_summary', 'R2', 'runtime', 'whiteness_p'); % Saving the data to be used in the subsequent main_ieeg(2) call.
            catch ME                                                        % For many of our data segments, in ill-conditioned matrix within the subpace ID method prevented it from continuing. We detect that specific error here and flag that segment to be skipped when main_ieeg(1) is run again to complete all segments.
                if isequal(ME.message, 'Input to SVD must not contain NaN or Inf.')
                    load skip_segments.mat skip_segments note
                    skip_segments(end+1) = i_segment;
                    skip_segments = sort(skip_segments, 'ascend');
                    note{end+1} = [num2str(i_segment) ': Same error thrown by linear_subspace as 967'];
                    save skip_segments.mat skip_segments note
                else
                    rethrow(ME)                                             % Rethrow the error if it was caused for any reason other than the above.
                end
            end
        end
    case 2                                                                  % Section 2: collecting the results of all jobs from main_ieeg(1). Must be run after completion of all jobs on the cluster/locally using main_ieeg(1). Any jobs not completed will be replaced with NaNs.
        R2_rec = cell(n_segment, 1);                                        % Cell array for collecting all R^2 vectors from methods.
        runtime_rec = cell(n_segment, 1);                                   % Same, but for the time that each method takes to run.
        whiteness_p_rec = cell(n_segment, 1);                               % Same, but for the vector of p values of chi-squared test of whiteness for each method.
        segment_done = false(n_segment, 1);                                 % Array of flags that indicates whether main_ieeg(1) each subject-CV fold has been completed and its data is avialable in main_data/
        
        for i_segment = 1:n_segment
            filename = ['main_data/' num2str(i_segment) '.mat'];            % The filename that should have been created in call to main_ieeg(1).
            if exist(filename, 'file')
                load(filename, 'R2', 'runtime', 'whiteness_p')
                R2_rec{i_segment} = R2;
                runtime_rec{i_segment} = runtime;
                whiteness_p_rec{i_segment} = whiteness_p;
                segment_done(i_segment) = true;
            end
        end
        
        save main_data.mat R2_rec runtime_rec whiteness_p_rec n_segment
    case 3                                                                  % Section 3: Plotting the results
        load main_data.mat R2_rec runtime_rec whiteness_p_rec
        n_method = size(R2_rec{1}, 2);
        R2_rec_2D = cell2mat(R2_rec);                                       % Flattening R2_rec so that each column contains combined data for all subjects and all brain regions for any given method. Same below.
        R2_rec_2D_for_plot = R2_rec_2D;
        min_R2_for_plot = 0.9;
        R2_rec_2D_for_plot(R2_rec_2D_for_plot < min_R2_for_plot - 1) = min_R2_for_plot - 1;
        whiteness_p_rec_2D = cell2mat(whiteness_p_rec);
        whiteness_p_rec_2D_for_plot = whiteness_p_rec_2D;
        min_whiteness_p_for_plot = 1e-100;
        whiteness_p_rec_2D_for_plot(whiteness_p_rec_2D_for_plot < min_whiteness_p_for_plot/10) = ...
            min_whiteness_p_for_plot/10;
        runtime_rec = cell2mat(runtime_rec);
        
        plot_ind = 1:n_method;                                              % List of methods to including in all plotting below.
        n_plot = numel(plot_ind);
        n_lin_method = 5;                                                   % Number of linear mehods
        n_nonlin_method = 4;                                                % Number of nonlinear methods
        colors = [0.3*ones(1, 3); repmat(matlab_green, n_lin_method, 1); repmat(matlab_yellow, n_nonlin_method, 1)]; % Colors of boxplots for each method
        labels = {'Zero', 'Linear\\[-2pt](dense)', 'Linear\\[-2pt](sparse)', 'AR-100\\[-2pt](sparse)', ...
            'AR-100\\[-2pt] (scalar)', 'Subspace', 'NMM', 'Manifold', 'DNN', 'MMSE\\[-2pt](scalar)'}; % Labels abbreviating each method
        labels = cellfun(@(label)['\parbox{4.5em}{\centering ' label '}'], labels, 'UniformOutput', 0); % Small modification for better latex rendering
        
        
        hf = figure;                                                        % The boxplot containing the R^2 comparisons
        hf.Position(3) = 640;
        boxplot(R2_rec_2D_for_plot(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', 1:n_plot, 'XTickLabelRotation', 90, ...
            'xticklabel', labels(plot_ind), 'fontsize', 20, 'TickLabelInterpreter', 'latex')
        ylim([0.9 1])
        ylabel('$R^2$', 'Interpreter', 'latex')
        xlim([0.5 n_plot+0.5])
        grid on
        export_fig main_ieeg_3_R2.eps -transparent
        
        figure                                                              % The lower-triangular matrix plot of the p-values from comparison between R^2 distributions
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = ranksum(R2_rec_2D(:, plot_ind(i)), R2_rec_2D(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        set(gcf, 'Color', 'w')
        export_fig main_ieeg_3_R2_p.eps

        
        hf = figure;                                                        % Same as main_ieeg_3_R2.eps but for p values of chi-squared test of whiteness
        hf.Position(3) = 640;
        boxplot(whiteness_p_rec_2D_for_plot(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        hold on
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', 1:numel(plot_ind), 'XTickLabelRotation', 90, ...
            'xticklabel', labels(plot_ind), 'fontsize', 20, 'yscale', 'log', ...
            'TickLabelInterpreter', 'latex')
        xlim([0.5 n_plot+0.5])
        xlims = get(gca, 'xlim');
        plot(xlims, [0.05 0.05], 'k--')
        ylabel('Whiteness p-value', 'Interpreter', 'latex')
        grid on
        xlim(xlims)
        ylim([1e-100 1])
        hf.Color = 'w';
        export_fig main_ieeg_3_p.eps
        
        figure                                                              % Same as main_ieeg_3_R2_p.eps but for p values of chi-squared test of whiteness
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = ranksum(whiteness_p_rec_2D(:, plot_ind(i)), whiteness_p_rec_2D(:, plot_ind(j)), ...
                    'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        set(gcf, 'Color', 'w')
        export_fig main_ieeg_3_p_p.eps
        
        
        hf = figure;                                                        % Same as main_ieeg_3_R2.eps but for run times
        hf.Position(3) = 640;
        boxplot(runtime_rec(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', 1:numel(plot_ind), 'XTickLabelRotation', 90, ...
            'xticklabel', labels(plot_ind), 'ytick', 10.^(-2:2:4), 'fontsize', 20, 'yscale', 'log', ...
            'TickLabelInterpreter', 'latex')
        ylabel('Run Time (seconds)', 'Interpreter', 'latex')
        xlim([0.5 n_plot+0.5])
        ylim([1e-2 1e5])
        grid on
        hf.Color = 'w';
        export_fig main_ieeg_3_time.eps
        
        figure                                                              % Same as main_ieeg_3_R2_p.eps but for run times
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = ranksum(runtime_rec(:, plot_ind(i)), runtime_rec(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        set(gcf, 'Color', 'w')
        export_fig main_ieeg_3_time_p.eps
end
end

%% Auxiliary functions
function write_main_sh                                                      % This function writes the shell script main.sh to the same directory. This is used for running jobs on a cluster. The main.sh could be directly written, but in this way all the code is accessible from MATLAB. Note that this is unnecessary if the jobs are going to be run locally.
s = ["#!/bin/bash";
    "matlab -r ""main_ieeg(1.5, $1); exit"""];
fileID = fopen('main.sh', 'w');
fprintf(fileID, '%s\n', s);
end

function plot_p_cmp(p, labels, correction, log10p_range)                    % Function for plotting a lower triangular matrix of p-values, with hot colors corresponding to entires (i, j), i > j, such that p(i, j) < p_thr, cold colors to entires (i, j), i > j, such that p(j, i) < p_thr, and gray hatches if neither of p(i, j) or p(j, i) is less than p_thr. p_thr is the corrected of 0.05 using either the Bonferroni or FDR correction. 
if nargin < 3 || isempty(correction)
    correction = 'FDR';
end
if nargin < 4 || isempty(log10p_range)
    log10p_range = [-10 0];
end
n_plot = size(p, 1);
alpha = 0.05;
n_pair = nchoosek(n_plot, 2);
switch correction
    case 'Bonferroni'
        p_thr = alpha / n_pair;
        sig = p < p_thr;
    case 'FDR'
        min_p = min(p, p');
        min_p_vec = sort(min_p(tril(true(n_plot), -1)), 'ascend');
        critical_vals = (1:n_pair)'/n_pair*alpha;
        max_p_ind = find(min_p_vec < critical_vals, 1, 'last');
        p_thr = critical_vals(max_p_ind);
        sig = p < p_thr;
end
ha1 = axes;
min_color1 = matlab_red;
mid_color1 = 1 - (1 - matlab_red) * 0.5;
max_color = ones(1, 3);
cmap1 = interp1([0 0.5 1], [min_color1; mid_color1; max_color], 0:0.001:1);
log10p_span = linspace(log10p_range(1), log10p_range(2), size(cmap1, 1));
cmap1(log10p_span > log10(p_thr), :) = 0.8;
min_color2 = matlab_blue;
mid_color2 = 1 - (1 - matlab_blue) * 0.5;
cmap2 = interp1([0 0.5 1], [min_color2; mid_color2; max_color], 0:0.001:1);
cmap2(log10p_span > log10(p_thr), :) = 0.8;
colormap(ha1, cmap1)
[sig_row, sig_col] = find(sig & tril(true(n_plot), -1));
n_sig = numel(sig_row);
for i_sig = 1:n_sig
    patch(sig_col(i_sig) + [-0.5 -0.5 0.5 0.5], sig_row(i_sig) + [-0.5 0.5 0.5 -0.5], ...
        interp1(log10p_span, cmap1, log10(p(sig_row(i_sig), sig_col(i_sig))), 'nearest', 'extrap'), ...
        'linestyle', 'none');
end
hc1 = colorbar;
hc1.Ticks = [];
caxis(log10p_range);
axis equal
axis([0.5 n_plot-0.5 1.5 n_plot+0.5])
set(gca, 'xtick', 1:n_plot-1, 'xticklabel', labels(1:end-1), 'ytick', 2:n_plot, ...
    'yticklabel', labels(2:end), 'xticklabelrotation', 90, 'ticklength', [0 0], ...
    'ticklabelinterpreter', 'latex', 'fontsize', 15, 'YDir', 'reverse')
[sig_row, sig_col] = find(sig' & tril(true(n_plot), -1));
n_sig = numel(sig_row);
for i_sig = 1:n_sig
    patch(sig_col(i_sig) + [-0.5 -0.5 0.5 0.5], sig_row(i_sig) + [-0.5 0.5 0.5 -0.5], ...
        interp1(log10p_span, cmap2, log10(p(sig_col(i_sig), sig_row(i_sig))), 'nearest', 'extrap'), ...
        'linestyle', 'none');
end
not_sig = ~sig & ~sig' & tril(true(n_plot), -1);
[not_sig_row, not_sig_col] = find(not_sig);
n_not_sig = nnz(not_sig);
for i_not_sig = 1:n_not_sig
    hp = patch(not_sig_col(i_not_sig) + [-0.5 -0.5 0.5 0.5], ...
        not_sig_row(i_not_sig) + [-0.5 0.5 0.5 -0.5], 'w', 'linestyle', 'none');
    hatchfill(hp, 'single', 45, 5, 0.95*[1 1 1]);
end
ha2 = axes;
ha2.Visible = 'off';
colormap(ha2, cmap2);
caxis([-10 0]);
hc2 = colorbar(ha2 ,'Position', hc1.Position+[hc1.Position(3) 0 0 0]);
hc2.Ticks = -10:2:0;
hc2.TickLabels = arrayfun(@(i)['$10^{' num2str(i) '}$'], linspace(log10p_range(1), log10p_range(end), 6), ...
    'UniformOutput', 0);
hc2.TickLabels{1} = ['$<10^{' num2str(log10p_range(1)) '}$'];
hc2.TickLabelInterpreter = 'latex';
hc2.FontSize = 15;
hc2.Label.String = '$p$';
hc2.Label.Interpreter = 'latex';
end