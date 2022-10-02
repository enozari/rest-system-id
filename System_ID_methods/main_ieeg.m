function main_ieeg(section, ss_factor, k, varargin)
%MAIN_IEEG The wrapper code for reproducing the iEEG analyses and graphics
% reported in the manuscript 
% E. Nozari et. al., "Is the brain macroscopically linear? A system
% identification of resting state dynamics", 2021.
%
%   section can take any of 4 values:
%       'submit': submit one job per subject-cross validation to the
%       cluster, each running all system identification methods on the
%       respective data
%       'run': running all methods on one subject-cross validation data
%       'gather': once all subject-cross validations are completed,
%       gathering the results
%       'plot': plotting the various figures reported in the above study
% 
%   ss_factor can take any of 3 values (to replicate the results of the
%   study): 1, 5, 25. The value of ss_factor determines the subsampling
%   factor used to down-sample the data. Note that this function does not
%   perform the downsampling. All the data are assumed to be downsampled
%   and available in the appropriate directories (see under section ==
%   'run' for the default directories). This input argument just determines
%   where to look for the data and how to name the generated files.
% 
%   k is the number of multi-step ahead predictions for cross-validation.
% 
%   Additional input arguments:
% 
%   if section == 'submit': varargin{1} = maximum number of jobs to be
%   simultaneously submitted to the cluster
% 
%   if section == 'run': varargin{1} = index of the time series segment
%   that should be loaded and used for analysis, varargin{2} = flag
%   indicating whether the subspace method should be run for this data
%   segment
% 
%   IMPORTANT NOTE: If running on cluster, this code assumes that this
%   function (main_ieeg) is already pre-compiled (using mcc -m main_ieeg.m)
%   to prevent license over-use. That would automatically create the
%   run_main_ieeg.sh shell code used below, and has to be repeated whenever
%   this function or any of the functions in its call chain is changed.
%   Note that for precompilation, all the required files (other packages,
%   etc.) must be in the same folder as this function (not just on MATLAB
%   path).
% 
%   Copyright (C) 2021, Erfan Nozari
%   All rights reserved.

if ischar(ss_factor)
    ss_factor = str2double(ss_factor);
end

run_on_cluster = 1;

if ~run_on_cluster
    % Adding the parent directory and its sub-directories to the path
    full_filename = mfilename('fullpath');
    slash_loc = strfind(full_filename, '/');
    addpath(genpath(full_filename(1:slash_loc(end))))
end

segments_address = ['rs_5min/rand_segments_' num2str(ss_factor)];
listing = struct2cell(dir(segments_address));
names = listing(1, :)';
n_segment = nnz(cellfun(@(name)name(1) ~= '.', names));

segment_sel_ratio = 0.1;                                                       % The ratio of subjects used for all the ensuing analyses.
segments = sort(randperm(n_segment, round(n_segment * segment_sel_ratio)), 'ascend');
n_segment = numel(segments);


switch section
    case 'submit'
        if ~exist('main_data', 'dir')
            mkdir main_data                                                 % Directory to store the created data files. The 'gather' section will later look inside this folder.
        end
        if ~exist('main_data_subspace', 'dir')
            mkdir main_data_subspace
        end 
        if exist(['nan_segments_' num2str(ss_factor) '.mat'], 'file')
            load(['nan_segments_' num2str(ss_factor) '.mat'], 'nan_segments')
        else
            nan_segments = [];
        end
        if exist(['skip_segments_' num2str(ss_factor) '.mat'], 'file')
            load(['skip_segments_' num2str(ss_factor) '.mat'], 'skip_segments')
        else
            skip_segments = [];
            save(['skip_segments_' num2str(ss_factor) '.mat'], 'skip_segments')
        end
        skip_segments = union(skip_segments, nan_segments);
        if ~exist('error_segments', 'dir')
            mkdir error_segments                                                 % Directory to store the created data files. The 'gather' section will later look inside this folder.
        end
        
        if ~isempty(varargin)
            max_jobs = varargin{1};                                         % The maximum number of jobs that should be submitted to the cluster at once or the number of parallel workers to be called if run locally.
        else
            max_jobs = inf;
        end
        
        if run_on_cluster
            
            write_main_sh();                                                    % Function to write the shell code main.sh for submitting SGE jobs.
            system('chmod +x main.sh');
            
            if exist('job_id_rec.mat', 'file')
                load job_id_rec.mat job_id_rec
            else
                job_id_rec = nan(0, 2);
            end
                        
            try
                [~, cmdout] = system('qstat');
            catch
                warning('Calling qstat in main failed, returning prematurely ...')
                return
            end
            newline_ind = strfind(cmdout, newline);
            if numel(newline_ind) > 2
                cmdout = cmdout(newline_ind(2)+1:end);
                newline_ind = strfind(cmdout, newline);
                cmdout = reshape(cmdout, [], numel(newline_ind))';
                job_id_active = str2num(cmdout(:, 1:7));
            else
                job_id_active = [];
            end
            
            load(['main_data_' num2str(ss_factor) '_pre.mat'], 'runtime_rec', 'R2_rec')
            skip = cellfun(@isempty, runtime_rec);
            runtime_rec = cell2mat(runtime_rec);
            n_rec = cellfun(@(R2)size(R2, 1), R2_rec(~skip));
            i_method = 6;                                                   % Corresponding to the subspace method in main_data_pre.mat
            p = polyfit(n_rec, runtime_rec(:, i_method), 1);
            job_id_del = [];
            for job_id = job_id_active'
                i_segment = job_id_rec(job_id_rec(:, 2) == job_id, 1);
                filename = ['main_data_subspace/' num2str(i_segment) '_t0.mat'];
                if exist(filename, 'file')
                    load(filename, 'subspace_t0', 'n')
                    if datetime - subspace_t0 > 10 * seconds(polyval(p, n))
                        job_id_del(end+1) = job_id;
                        delete(filename)
                        fileID = fopen(['main_data_subspace/' num2str(i_segment) '_skip'], 'w');
                        fprintf(fileID, '', '');
                    end
                end
            end
            if ~isempty(job_id_del)
                system(['qdel ' num2str(job_id_del)]);
            end
            job_id_active = setdiff(job_id_active, job_id_del);

            job_id_rec_isactive = ismember(job_id_rec(:, 2), job_id_active);
            
            if exist(['crash_segments_' num2str(ss_factor) '.mat'], 'file')
                load(['crash_segments_' num2str(ss_factor) '.mat'], 'crash_segments')
            else
                crash_segments = [];
            end
            inactive_segments = job_id_rec(~job_id_rec_isactive, 1);
            for i_segment = inactive_segments'
                if exist(['main_data_subspace/' num2str(i_segment) '_t0.mat'], 'file') ... &&
                        && ~exist(['main_data/' num2str(i_segment) '.mat'], 'file')
                    crash_segments(end+1) = i_segment;
                end
            end
            save(['crash_segments_' num2str(ss_factor) '.mat'], 'crash_segments')
            
            job_id_rec = job_id_rec(job_id_rec_isactive, :);
            
            if ~exist('trash', 'dir')
                mkdir trash
            end
            [~, cmdout] = system('find main.sh.o*');
            newline_ind = strfind(cmdout, newline);
            cmdout = reshape(cmdout, newline_ind(1), [])';
            job_id_inactive = setdiff(str2num(cmdout(:, 10:16)), job_id_active);
            for i = 1:numel(job_id_inactive)
                system(['mv main.sh.*' num2str(job_id_inactive(i)) ' trash']);
            end
            
            n_jobs = 0;
            for i_segment = 1:n_segment
                if ~exist(['main_data/' num2str(i_segment) '.mat'], 'file') ... && 
                        && ~ismember(i_segment, job_id_rec(:, 1)) && ~ismember(i_segment, skip_segments) ... &&
                        && n_jobs < max_jobs - numel(job_id_active)
                    run_subspace = ~exist(['main_data_subspace/' num2str(i_segment) '_skip'], 'file') ... &&
                        && ~any(crash_segments == i_segment);
                    [~, ~] = system('rm -f /cbica/home/nozarie/.matlab/R2018a/toolbox_cache-9.4.0-3284594471-glnxa64.xml');
                    [~, cmdout] = system(['qsub ./main.sh ' num2str(ss_factor) ' ' num2str(k) ' ' num2str(i_segment) ' ' num2str(run_subspace)]) 
                    job_id_rec(end+1, :) = [i_segment, str2double(cmdout(strfind(cmdout, 'Your job ')+(9:15)))];
                    n_jobs = n_jobs + 1;
                end
            end
            
            save job_id_rec.mat job_id_rec
        else
            hp = gcp('nocreate');
            if isempty(hp)
                parpool(max_jobs)
            elseif hp.NumWorkers ~= max_jobs
                delete(hp)
                parpool(max_jobs)
            end
            parfor_progress(n_segment);
            parfor i_segment = 1:n_segment
                if ismember(i_segment, skip_segments) ... ||
                        || exist(['main_data/' num2str(i_segment) '.mat'], 'file')
                    continue
                end
                run_subspace = ~exist(['main_data_subspace/' num2str(i_segment) '_skip'], 'file');
                main_ieeg('run', ss_factor, k, i_segment, run_subspace)
                parfor_progress;
            end
            parfor_progress(0);
        end
        
    case 'run'
        if ischar(k)
            k = str2double(k);
        end
        i_segment = varargin{1};
        run_subspace = varargin{2};
        if ischar(i_segment)
            i_segment = str2double(i_segment);
        end
        if ischar(run_subspace)
            run_subspace = str2double(run_subspace);
        end
        
        load([segments_address '/Y_' num2str(segments(i_segment)) '.mat'], 'Y')
        if any(isnan(Y(:)))
            if exist(['nan_segments_' num2str(ss_factor) '.mat'], 'file')
                load(['nan_segments_' num2str(ss_factor) '.mat'], 'nan_segments')
            else
                nan_segments = [];
            end
            nan_segments(end+1) = i_segment;
            nan_segments = unique(nan_segments);
            save(['nan_segments_' num2str(ss_factor) '.mat'], 'nan_segments')
        else
            try
                MMSE_memory = -64;
                warning('off')
                [model_summary, R2, runtime, whiteness, ~, Y_hat] = ...
                    all_methods_ieeg(Y, [], k, [], run_subspace, i_segment, MMSE_memory);          % Calling the routine that runs all system id methods.
                warning('on')
                save(['main_data/' num2str(i_segment) '.mat'], 'model_summary', 'R2', 'runtime', 'whiteness', 'Y_hat'); % Saving the data to be used in the subsequent 'gather' section.
            catch ME
                if isequal(ME.message, 'Input to SVD must not contain NaN or Inf.')
                    load(['skip_segments_' num2str(ss_factor) '.mat'], 'skip_segments')
                    skip_segments(end+1) = i_segment;
                    skip_segments = sort(skip_segments, 'ascend');
                    save(['skip_segments_' num2str(ss_factor) '.mat'], 'skip_segments')
                else
                    if ~isequal(ME.identifier, 'MATLAB:license:checkouterror')
                        save(['error_segments/' num2str(i_segment) '.mat'], 'ME')
                    end
                    warning(['ERROR ENCOUNTERED IN i_segment = ' num2str(i_segment) ' ' repmat('=', 1, 80)])
                end
            end
        end
        
    case 'gather'
        R2_rec = cell(n_segment, 1);                                        % Cell array for collecting all R^2 vectors from methods.
        runtime_rec = cell(n_segment, 1);                                   % Same, but for the time that each method takes to run.
        whiteness_rec = cell(n_segment, 1);
        segment_done = false(n_segment, 1);
        
        for i_segment = 1:n_segment
            filename = ['main_data/' num2str(i_segment) '.mat'];            % The filename that should have been created in call to the 'submit' section.
            if exist(filename, 'file')
                load(filename, 'R2', 'runtime', 'whiteness')
                R2_rec{i_segment} = R2;
                runtime_rec{i_segment} = runtime;
                whiteness_rec{i_segment} = whiteness;
                segment_done(i_segment) = true;
            end
        end
        whiteness_rec = cell2mat(whiteness_rec(segment_done));
        
        save(['main_data_' num2str(ss_factor) '_k' num2str(k) '.mat'], 'R2_rec', 'runtime_rec', 'whiteness_rec', 'n_segment', 'segment_done')
    
    case 'plot'
        load(['main_data_' num2str(ss_factor) '_k' num2str(k) '.mat'], 'R2_rec', 'runtime_rec', 'whiteness_rec')
        n_method = size(R2_rec{1}, 2);
        R2_rec_2D = cell2mat(R2_rec);                                       % Flattening R2_rec so that each column contains combined data for all subjects and all brain regions for any given method. Same below.
        whiteness_stat_rec_2D = reshape([whiteness_rec.stat], [], n_method);
        whiteness_sig_thr_rec_2D = reshape([whiteness_rec.sig_thr], [], n_method);
        whiteness_rel_stat_rec_2D = whiteness_stat_rec_2D ./ whiteness_sig_thr_rec_2D;
        runtime_rec = cell2mat(runtime_rec);
        
        plot_ind = 1:n_method;                                              % List of methods to including in all plotting below. Uninteresting methods are taken out.
        n_plot = numel(plot_ind);
        n_lin_method = 5;
        n_nonlin_method = 7;
        colors = [0.3*ones(1, 3); repmat(matlab_green, n_lin_method, 1); repmat(matlab_yellow, n_nonlin_method, 1)]; % Colors of boxplots for each method
        labels = {'Zero';
            'Linear\\[-2pt](dense)';
            'Linear\\[-2pt](sparse)';
            'AR-100\\[-2pt](sparse)';
            'AR-100\\[-2pt](scalar)';
            'Subspace';
            'NMM';
            'Manifold';
            'DNN\\[-2pt] (MLP)';
            'DNN\\[-2pt] (CNN)';
            'LSTM\\[-2pt] (IIR)';
            'LSTM\\[-2pt] (FIR)';
            'MMSE\\[-2pt](scalar)'};                                        % Labels abbreviating each method
        labels = cellfun(@(label)['\parbox{4.5em}{\centering ' label '}'], labels, 'UniformOutput', 0); % Small modification for better latex rendering
        
        
        hf = figure;                                                        % The boxplot containing the R^2 comparisons
        hf.Position(3) = 640;
        boxplot(R2_rec_2D(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', 1:n_plot, 'XTickLabelRotation', 90, ...
            'xticklabel', labels(plot_ind), 'fontsize', 15, 'TickLabelInterpreter', 'latex')
        switch ss_factor
            case 1
                ylim([0.9 1])
            case 5
                ylim([0.5 1])
            case 25
                ylim([0 1])
        end
        ylabel('$R^2$', 'Interpreter', 'latex')
        xlim([0.5 n_plot+0.5])
        grid on
        if ss_factor == 1
            exportgraphics(hf, ['main_ecog_3_R2_k' num2str(k) '.eps'])
        else
            exportgraphics(hf, ['main_ecog_3_R2_' num2str(ss_factor) '_k' num2str(k) '.eps'])
        end
        
        figure                                                              % The lower-triangular matrix plot of the p-values from comparison between R^2 distributions
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = signrank(R2_rec_2D(:, plot_ind(i)), R2_rec_2D(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        set(gcf, 'Color', 'w')
        if ss_factor == 1
            exportgraphics(gcf, ['main_ecog_3_R2_p_k' num2str(k) '.eps'])
        else
            exportgraphics(gcf, ['main_ecog_3_R2_p_' num2str(ss_factor) '_k' num2str(k) '.eps'])
        end
        
        
        hf = figure;
        hf.Position(3) = 640;
        boxplot(whiteness_rel_stat_rec_2D(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', 1:numel(plot_ind), 'XTickLabelRotation', 90, ...
            'xticklabel', labels(plot_ind), 'fontsize', 15, 'yscale', 'log', ...
            'TickLabelInterpreter', 'latex')
        xlim([0.5 n_plot+0.5])
        ylabel('Whiteness Statistic ($Q / Q_{\rm thr}$)', 'Interpreter', 'latex')
        grid on
        hf.Color = 'w';
        if ss_factor == 1
            exportgraphics(hf, ['main_ecog_3_p_k' num2str(k) '.eps'])
        else
            exportgraphics(hf, ['main_ecog_3_p_' num2str(ss_factor) '_k' num2str(k) '.eps'])
        end
        
        figure
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = signrank(whiteness_rel_stat_rec_2D(:, plot_ind(i)), ...
                    whiteness_rel_stat_rec_2D(:, plot_ind(j)), ...
                    'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        set(gcf, 'Color', 'w')
        if ss_factor == 1
            exportgraphics(gcf, ['main_ecog_3_p_p_k' num2str(k) '.eps'])
        else
            exportgraphics(gcf, ['main_ecog_3_p_p_' num2str(ss_factor) '_k' num2str(k) '.eps'])
        end
        
        
        hf = figure;
        hf.Position(3) = 640;
        boxplot(runtime_rec(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', 1:numel(plot_ind), 'XTickLabelRotation', 90, ...
            'xticklabel', labels(plot_ind), 'ytick', 10.^(-2:2:4), 'fontsize', 15, ...
            'yscale', 'log', 'TickLabelInterpreter', 'latex')
        ylabel('Run Time (s)', 'Interpreter', 'latex')
        xlim([0.5 n_plot+0.5])
        ylim([1e-2 1e5])
        grid on
        hf.Color = 'w';
        if ss_factor == 1
            exportgraphics(hf, ['main_ecog_3_time_k' num2str(k) '.eps'])
        else
            exportgraphics(hf, ['main_ecog_3_time_' num2str(ss_factor) '_k' num2str(k) '.eps'])
        end
        
        figure
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = signrank(runtime_rec(:, plot_ind(i)), runtime_rec(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        set(gcf, 'Color', 'w')
        if ss_factor == 1
            exportgraphics(gcf, ['main_ecog_3_time_p_k' num2str(k) '.eps'])
        else
            exportgraphics(gcf, ['main_ecog_3_time_p_' num2str(ss_factor) '_k' num2str(k) '.eps'])
        end
        
        % Comparison p-values using t-test instead of signrank
        figure
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                [~, p(i, j)] = ttest(R2_rec_2D(:, plot_ind(i)), R2_rec_2D(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        exportgraphics(gcf, 'main_ecog_3_R2_p_ttest.eps')
        
        figure
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                [~, p(i, j)] = ttest(whiteness_rel_stat_rec_2D(:, plot_ind(i)), ...
                    whiteness_rel_stat_rec_2D(:, plot_ind(j)), ...
                    'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        exportgraphics(gcf, 'main_ecog_3_p_p_ttest.eps')
        
        figure
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                [~, p(i, j)] = ttest(runtime_rec(:, plot_ind(i)), runtime_rec(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        exportgraphics(gcf, 'main_ecog_3_time_p_ttest.eps')
end
end

%% Auxiliary functions
function write_main_sh                                                 % This function writes the shell script main.sh to the same directory. This is used for running jobs on a cluster. The main.sh could be directly written, but in this way all the code is accessible from MATLAB. Note that this is unnecessary if the jobs are going to be run locally. The shell code run_main_ieeg.sh should already be present in the same directory by running mcc -m main_ieeg.m in MATLAB.
s = ["#! /bin/bash";
    "#$ -S /bin/bash";
    "#$ -pe threaded 2";
    "#$ -l h_vmem=64G";
    "#$ -l s_vmem=64G";
    "./run_main_ieeg.sh $MATLAB_DIR run $1 $2 $3 $4"];
fileID = fopen('main.sh', 'w');
fprintf(fileID, '%s\n', s);
end

function plot_p_cmp(p, labels, correction, log10p_range)                                              % Function for plotting a lower triangular matrix of p-values, with hot colors corresponding to entires (i, j), i > j, such that p(i, j) < 0.05, cold colors to entires (i, j), i > j, such that p(j, i) < 0.05, and gray hatches if neither of p(i, j) or p(j, i) is less than 0.05. 
if nargin < 3 || isempty(correction)
    correction = 'FDR';
end
if nargin < 4 || isempty(log10p_range)
    log10p_range = [-20 0];
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
    'ticklabelinterpreter', 'latex', 'fontsize', 13, 'YDir', 'reverse')
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