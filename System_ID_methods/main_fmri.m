function main_fmri(section, varargin)
%MAIN_FMRI The wrapper code for reproducing the analyses and graphics
% reported in the manuscript 
% E. Nozari et. al., "Is the brain macroscopically linear? A system
% identification of resting state dynamics", 2020.
%
%   main_fmri(section), where section = 1, 1.5, 2, or 3 runs the
%   corresponding section of the script. The sections are supposed to be
%   run in order, and each one should only be run if the previous sections
%   have been completed. Only sections 1, 2, and 3 are for external call,
%   whereas section 1.5 is internally called by section 1.
% 
%   main_fmri(section, max_jobs) with section = 1 optionally determines the
%   maximum number of jobs to be submitted to the cluster at any point in
%   time. This is ignored if run_on_cluster = 0.
% 
%   As stated above, the workflow is as follows: first run main_fmri(1),
%   and wait until it finishes running. Since this typically takes a large
%   amount of time, it is by default assumed that the code is run on a
%   cluster with Sun Grid Engine (SGE) job scheduler. If not, set
%   run_on_cluster = 0. When compelete, run main_fmri(2) to collect the
%   results of main_fmri(1) and save them in main_data.mat. When complete,
%   run main_fmri(3) to plot the graphics reported in the paper.
% 
%   Note that when the code is run on cluster with limited number of MATLAB
%   licenses, max_jobs should be set to as many MATLAB licenses as one can
%   use. If so, then main_fmri(1, max_jobs) needs to be called over and
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

load subjs.mat subjs                                                        % subjs is a cell array of HCP subjects used for the study.
n_subj = numel(subjs);
n_cv = 8;                                                                   % Number of cross-validation folds.
switch section
    case 1                                                                  % Section 1: submitting SGE jobs (or running them locally), one per subject per cross validation fold.
        if ~exist('main_data', 'dir')
            mkdir main_data                                                 % Directory to store the created data files. main_fmri(2) will later look inside this folder to gather results.
        end
        if ~exist('main_data_dnn', 'dir')                                   % Directory to keep track of which jobs have run to the point of running the DNN method. This is important since the DNN trainer occationally hangs indefinitely, and the job needs to be killed and restarted (done automatically by the code below).
            mkdir main_data_dnn
        end 
        if run_on_cluster
            if ~isempty(varargin)
                max_jobs = varargin{1};                                     % If a second optional parameter is provided in calling main_fmri(1, max_jobs), it indicates the maximum amount of jobs that should be submitted to the cluster at once. 
            else
                max_jobs = inf;
            end
            write_main_sh();                                                % Function to write the shell code main.sh for submitting SGE jobs.
            system('chmod +x main.sh');                                     % Making the shell code generated by the line above executable by the shell.
            
            if exist('job_id_rec.mat', 'file')
                load job_id_rec.mat job_id_rec                              % job_id_rec is a record of all the jobs that have been actively running on the cluster since the last call to this function
            else
                job_id_rec = cell(0, 2);
            end
            try
                [~, cmdout] = system('qstat');                              % Getting the latest stats of which jobs are running on the cluster
            catch 
                warning('Calling qstat in main failed, returning prematurely ...')
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
            
            job_id_del = [];                                                % List of jobs to be deleted, because the DNN trainer has not returned for at least 2 minutes. Adjust this time for your own dataset.
            for job_id = job_id_active'
                subj_i_cv = job_id_rec{[job_id_rec{:, 2}] == job_id, 1};
                filename = ['main_data_dnn/' subj_i_cv '_dnn.mat'];
                if exist(filename, 'file')
                    load(filename, 'dnn_dt')
                    if datetime - dnn_dt > minutes(2)
                        job_id_del(end+1) = job_id;
                        delete(filename)
                    end
                end
            end
            if ~isempty(job_id_del)
                system(['qdel ' num2str(job_id_del)]);                      % Killing the jobs in job_id_del from the cluster
            end
            job_id_active = setdiff(job_id_active, job_id_del);
            
            job_id_rec_isactive = ismember([job_id_rec{:, 2}], job_id_active);
            job_id_rec = job_id_rec(job_id_rec_isactive, :);                % Updating job_id_rec by removing the jobs that have either completed since the last call or killed above
            
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
            for i_subj = 1:n_subj
                subj = subjs{i_subj};
                for i_cv = 1:n_cv
                    if ~exist(['main_data/' subj '_' num2str(i_cv) '.mat'], 'file') ... &&
                            && ~ismember([subj '_' num2str(i_cv)], job_id_rec(:, 1)) ... &&
                            && n_jobs < max_jobs - numel(job_id_active)     % First it checks whether that subject-CV fold combination has been completed before. This is useful if max_jobs is not infinity, so main_fmri(1, max_jobs) needs to be called multiple times. Then it checks if that job is currently being run on a node, and finally if there is room for submitting new jobs.
                        [~, cmdout] = system(['qsub -l s_vmem=16G -l h_vmem=16G ./main.sh ' subj ' ' num2str(i_cv)]); % Submitting the job to the cluster
                        job_id_rec(end+1, :) = {[subj '_' num2str(i_cv)], ...
                            str2double(cmdout(strfind(cmdout, 'Your job ')+(9:15)))}; % Keeping record of the just-submitted job
                        n_jobs = n_jobs + 1;
                    end
                end
            end
            
            save job_id_rec.mat job_id_rec
        else                                                                % If run_on_cluster = 0, the jobs are serially run on the current machine until complete.
            for i_subj = 1:n_subj
                subj = subjs{i_subj};
                for i_cv = 1:n_cv
                    main_fmri(1.5, subj, i_cv)
                end
            end
        end
    case 1.5                                                                % Section 1.5: the "inner" script running all methods for each subject-cross validation fold. This is only to be called internally by main_fmri(1), not by the user.
        subj = varargin{1};
        i_cv = varargin{2};
        scans = {'1_LR', '1_RL', '2_LR', '2_RL'};                           % The set of IDs for the four resting state scans that each subject has under the HCP protocol.
        n_scan = numel(scans);
        Y = cell(1, n_scan);                                                % A cell array of data segments used for system identification. Each element of Y is one scan, with channels along the first dimension and time along the second dimension.
        Y_pw = cell(1, n_scan);
        for i_scan = 1:n_scan
            scan = scans{i_scan};
            Y{i_scan} = readNPY(['HCP/yeo_100_' subj '_REST' scan '.npy']); % Reading the pre-processed and parcellated time series for each scan. Change this to your data directory and filename if different.
            Y_pw{i_scan} = Y{i_scan}(:, end/4+1:end/2);                     % Using only the second quarter of each scan for pairwise estimates to reduce computational complexity. Significantly less data is needed as well anyways in 2 dimensions vs. 116!
        end
        TR = 0.72;                                                          % The sampling time of HCP. Change to the sampling time of your dataset if different.
        test_range = [(i_cv-1)/n_cv, i_cv/n_cv];                            % A sub-interval of [0, 1] indicating the portion of the data that is used for test (cross-validation). The rest of the data is used for training.
        MMSE_memory = -16;                                                  % Assuming 16GB of memory. See MMSE_est.m for details. Change to minus the GB of available memory if different.
        [model_summary, R2, R2_pw, runtime, whiteness_p, whiteness_p_pw] = ...
            all_methods_fmri(Y, Y_pw, TR, test_range, MMSE_memory, [subj '_' num2str(i_cv)]);          % Calling the routine that runs all system id methods.

        save(['main_data/' subj '_' num2str(i_cv) '.mat'], 'model_summary', 'R2', 'R2_pw', 'runtime', ...
            'whiteness_p', 'whiteness_p_pw');                               % Saving the data to be used in the subsequent main_fmri(2) call.
        
    case 2                                                                  % Section 2: collecting the results of all jobs from main_fmri(1). Must be run after completion of all jobs on the cluster/locally using main_fmri(1). Any jobs not completed will be replaced with NaNs.
        R2_rec = cell(1, 1, n_subj);                                        % Cell array for collecting all R^2 vectors from methods.
        R2_pw_rec = cell(1, 1, 1, n_subj);                                  % Same, but for pairwise methods (pairwise linear and pairwise MMSE).
        runtime_rec = cell(n_subj, 1);                                      % Same, but for the time that each method takes to run.
        whiteness_p_rec = cell(1, 1, n_subj);                               % Same, but for the vector of p values of chi-squared test of whiteness for each method.
        whiteness_p_pw_rec = cell(1, 1, 1, n_subj);                         % Same, but for pairwise methods.
        subj_cv_done = false(n_subj, n_cv);                                 % Array of flags that indicates whether main_fmri(1) for each subject-CV fold has been completed and its data is avialable in main_data/
        
        for i_subj = 1:n_subj
            subj = subjs{i_subj};
            n = 116;                                                        % Number of parcells
            n_method = 13;                                                  % Total number of system id methods used (not including pairwise methods)
            n_method_pw = 2;                                                % Number of pairwise methods run. This and the above 2 lines must be consistent with all_methods_fmri.m.
            R2_subj_rec = nan(n, n_method, n_cv);                           % The entry of R2_rec corresponding to subject subj. The same for other arrays below.
            R2_pw_subj_rec = nan(n, n, n_method_pw, n_cv);
            runtime_subj_rec = nan(n_cv, n_method+n_method_pw);
            whiteness_p_subj_rec = nan(n, n_method, n_cv);
            whiteness_p_pw_subj_rec = nan(n, n, n_method_pw, n_cv);
            for i_cv = 1:n_cv
                filename = ['main_data/' subj '_' num2str(i_cv) '.mat'];    % The filename that should have been created in call to main_fmri(1).
                if exist(filename, 'file')
                    load(filename, 'R2', 'R2_pw', 'runtime', 'whiteness_p', 'whiteness_p_pw')
                    R2_subj_rec(:, :, i_cv) = R2;
                    R2_pw_subj_rec(:, :, :, i_cv) = R2_pw;
                    runtime_subj_rec(i_cv, :) = runtime;
                    whiteness_p_subj_rec(:, :, i_cv) = whiteness_p;
                    whiteness_p_pw_subj_rec(:, :, :, i_cv) = whiteness_p_pw;
                    subj_cv_done(i_subj, i_cv) = true;
                end
            end
            R2_rec{i_subj} = mean(R2_subj_rec, 3, 'omitnan');               % Averaging over cross-validation folds, same for the subsequent four lines.
            R2_pw_rec{i_subj} = mean(R2_pw_subj_rec, 4, 'omitnan');
            runtime_rec{i_subj} = mean(runtime_subj_rec, 'omitnan');
            whiteness_p_rec{i_subj} = mean(whiteness_p_subj_rec, 3, 'omitnan');
            whiteness_p_pw_rec{i_subj} = mean(whiteness_p_pw_subj_rec, 4, 'omitnan');
        end
        R2_rec = cell2mat(R2_rec);                                          % Transforming from cell array to numerical array that takes less memory. Same below. This is n x n_method x n_subj.
        R2_pw_rec = cell2mat(R2_pw_rec);                                    % n x n_method_pw x n_subj
        runtime_rec = cell2mat(runtime_rec);                                % n_subj x (n_method + n_method_pw)
        whiteness_p_rec = cell2mat(whiteness_p_rec);                        % n x n_method x n_subj
        whiteness_p_pw_rec = cell2mat(whiteness_p_pw_rec);                  % n x n_method_pw x n_subj
        
        save main_data.mat R2_rec R2_pw_rec runtime_rec whiteness_p_rec whiteness_p_pw_rec subjs n_cv
    case 3                                                                  % Section 3: Plotting the results
        load main_data.mat R2_rec R2_pw_rec runtime_rec whiteness_p_rec whiteness_p_pw_rec
        [n, n_method, n_subj] = size(R2_rec);
        R2_rec_2D = reshape(permute(R2_rec, [2 1 3]), n_method, n * n_subj)'; % Flattening R2_rec so that each column contains combined data for all subjects and all brain regions for any given method. Same below.
        n_method_pw = size(R2_pw_rec, 3);
        R2_pw_rec_2D = cell2mat(arrayfun(@(i_method)reshape(R2_pw_rec(:, :, i_method, :), [], 1), ...
            1:n_method_pw, 'UniformOutput', 0));
        whiteness_p_rec_2D = reshape(permute(whiteness_p_rec, [2 1 3]), n_method, n * n_subj)';
        whiteness_p_pw_rec_2D = cell2mat(arrayfun(@(i_method)reshape(whiteness_p_pw_rec(:, :, i_method, :), ...
            [], 1), 1:n_method_pw, 'UniformOutput', 0));
        
        plot_ind = setdiff(1:n_method, [4 5 7]);                            % List of methods (not pairwise methods) to including in all plotting below. Any methods not to be plotted are taken out.
        n_plot = numel(plot_ind);
        plot_loc_pw = numel(plot_ind)+(2:3);                                % To be able to plot boxplots of brain-wise and pairwise methods side by side in a single axis
        n_lin_method = 8;
        n_nonlin_method = 4;
        colors = [0.3*ones(1, 3); repmat(matlab_green, n_lin_method, 1); ...
            repmat(matlab_yellow, n_nonlin_method, 1)];                     % Colors of boxplots for each method
        colors_pw = [matlab_green; matlab_yellow];
        labels = {'Zero', 'Linear\\[-2pt](dense)', 'Linear\\[-2pt](sparse)', 'VAR-2\\[-2pt](sparse)', ...
            'AR-2\\[-2pt](sparse)', 'VAR-3\\[-2pt](sparse)', 'AR-3\\[-2pt](sparse)', 'Linear\\[-3pt]w/ HRF', ...
            'Subspace', 'NMM', 'NMM\\[-2pt]w/ HRF', 'Manifold', 'DNN'};     % Labels abbreviating each method
        labels = cellfun(@(label)['\parbox{4.5em}{\centering ' label '}'], labels, 'UniformOutput', 0); % Small modification for better latex rendering
        labels_pw = {'Linear\\[-2pt](pairwise)', 'MMSE\\[-2pt](pairwise)'};
        labels_pw = cellfun(@(label)['\parbox{4.5em}{\centering ' label '}'], labels_pw, 'UniformOutput', 0);
        
%         % --- Methods figures
        hf = figure;                                                        % The boxplot containing the R^2 comparisons
        hf.Position(3) = 640;
        boxplot(R2_rec_2D(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        hold on
        boxplot(R2_pw_rec_2D, 'Positions', plot_loc_pw, 'Whisker', inf, 'Colors', colors_pw, 'Widths', 0.5)
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', [1:n_plot plot_loc_pw], 'XTickLabelRotation', 90, ...
            'xticklabel', [labels(plot_ind), labels_pw], 'fontsize', 15, 'TickLabelInterpreter', 'latex')
        ylims = get(gca, 'ylim');
        ylim([ylims(1) 1])
        ylabel('$R^2$', 'Interpreter', 'latex')
        xlim([0.5 plot_loc_pw(end)+0.5])
        grid on
        export_fig main_fmri_3_R2.eps -transparent
        
        figure                                                              % The lower-triangular matrix plot of the p-values from comparison between R^2 distributions
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = ranksum(R2_rec_2D(:, plot_ind(i)), R2_rec_2D(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        set(gcf, 'Color', 'w')
        export_fig main_fmri_3_R2_p.eps
        
        hf = figure;                                                        % Same as above but for pairwise methods (together with the zero method, which is taken from the R2_rec)
        hf.Position(3:4) = hf.Position(3:4) - 270;
        p = nan(n_method_pw+1);
        R2_pw_rec_2D = [{R2_rec_2D(:, 1)} fliplr(num2cell(R2_pw_rec_2D, 1))];
        for i = 1:n_method_pw+1
            for j = 1:n_method_pw+1
                p(i, j) = ranksum(R2_pw_rec_2D{i}, R2_pw_rec_2D{j}, 'tail', 'right');
            end
        end
        plot_p_cmp(p, [labels(1) fliplr(labels_pw)])
        hf.Children(1).Ticks(2:end-1) = [];
        hf.Children(1).TickLabels(2:end-1) = [];
        hf.Color = 'w';
        export_fig main_fmri_3_R2_pw_p.eps

        hf = figure;                                                        % Same as main_fmri_3_R2.eps but for p values of chi-squared test of whiteness
        hf.Position(3) = 640;
        boxplot(whiteness_p_rec_2D(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        hold on
        boxplot(whiteness_p_pw_rec_2D, 'Whisker', inf, 'Positions', plot_loc_pw, 'Colors', colors_pw, ...
            'Widths', 0.5)
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', [1:numel(plot_ind) plot_loc_pw], 'XTickLabelRotation', 90, ...
            'xticklabel', [labels(plot_ind), labels_pw], 'fontsize', 15, 'yscale', 'log', ...
            'TickLabelInterpreter', 'latex')
        xlims = [0.5 plot_loc_pw(end)+0.5];
        plot(xlims, [0.05 0.05], 'k--')
        ylabel('Whiteness p-value', 'Interpreter', 'latex')
        grid on
        xlim(xlims)
        ylim([1e-10 1])
        hf.Color = 'w';
        export_fig main_fmri_3_p.eps
        
        figure                                                              % Same as main_fmri_3_R2_p.eps but for p values of chi-squared test of whiteness
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = ranksum(whiteness_p_rec_2D(:, plot_ind(i)), whiteness_p_rec_2D(:, plot_ind(j)), ...
                    'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        set(gcf, 'Color', 'w')
        export_fig main_fmri_3_p_p.eps
        
        hf = figure;                                                        % Same as main_fmri_3_R2_pw_p.eps but for p values of chi-squared test of whiteness
        hf.Position(3:4) = hf.Position(3:4) - 270;
        p = nan(n_method_pw+1);
        whiteness_p_pw_rec_2D = [{whiteness_p_rec_2D(:, 1)} fliplr(num2cell(whiteness_p_pw_rec_2D, 1))];
        for i = 1:n_method_pw+1
            for j = 1:n_method_pw+1
                p(i, j) = ranksum(whiteness_p_pw_rec_2D{i}, whiteness_p_pw_rec_2D{j}, 'tail', 'right');
            end
        end
        plot_p_cmp(p, [labels(1) fliplr(labels_pw)])
        hf.Children(1).Ticks(2:end-1) = [];
        hf.Children(1).TickLabels(2:end-1) = [];
        hf.Color = 'w';
        export_fig main_fmri_3_p_pw_p.eps
        
        hf = figure;                                                        % Same as main_fmri_3_R2.eps but for run times
        hf.Position(3) = 640;
        boxplot(runtime_rec(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        hold on
        boxplot(runtime_rec(:, end-1:end), 'Whisker', inf, 'Positions', plot_loc_pw, 'Colors', colors_pw, ...
            'Widths', 0.5)
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', [1:numel(plot_ind) plot_loc_pw], 'XTickLabelRotation', 90, ...
            'xticklabel', [labels(plot_ind), labels_pw], 'ytick', 10.^(-1:4), 'fontsize', 15, 'yscale', 'log', ...
            'TickLabelInterpreter', 'latex')
        ylabel('Run Time', 'Interpreter', 'latex')
        xlim([0.5 plot_loc_pw(end)+0.5])
        ylim([1e-1 1e4])
        grid on
        hf.Color = 'w';
        export_fig main_fmri_3_time.eps
        
        figure                                                              % Same as main_fmri_3_R2_p.eps but for run times
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = ranksum(runtime_rec(:, plot_ind(i)), runtime_rec(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind))
        set(gcf, 'Color', 'w')
        export_fig main_fmri_3_time_p.eps
        
        hf = figure;                                                        % Same as main_fmri_3_R2_pw_p.eps but for run times
        hf.Position(3:4) = hf.Position(3:4) - 270;
        p = nan(n_method_pw+1);
        runtime_pw_rec = [{runtime_rec(:, 1)} fliplr(num2cell(runtime_rec(:, end-1:end), 1))];
        for i = 1:n_method_pw+1
            for j = 1:n_method_pw+1
                p(i, j) = ranksum(runtime_pw_rec{i}, runtime_pw_rec{j}, 'tail', 'right');
            end
        end
        plot_p_cmp(p, [labels(1) fliplr(labels_pw)])
        hf.Children(1).Ticks(2:end-1) = [];
        hf.Children(1).TickLabels(2:end-1) = [];
        hf.Color = 'w';
        export_fig main_fmri_3_time_pw_p.eps
        
        % --- Linear methods figures (for Supplementary Figures). This subsection produces figures in parallel to those above, but only for linear methods, as repored the supplementary
        plot_ind = 1:7;
        n_plot = numel(plot_ind);
        
        hf = figure;
        hf.Position(3) = 640;
        boxplot(R2_rec_2D(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', 1:n_plot, 'XTickLabelRotation', 90, ...
            'xticklabel', labels(plot_ind), 'fontsize', 25, 'TickLabelInterpreter', 'latex')
        ylims = get(gca, 'ylim');
        ylim([ylims(1) 1])
        ylabel('$R^2$', 'Interpreter', 'latex')
        xlim([0.5 n_plot+0.5])
        grid on
        export_fig main_fmri_3_R2_ARs.eps -transparent
        
        figure
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = ranksum(R2_rec_2D(:, plot_ind(i)), R2_rec_2D(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind), [], 20)
        set(gcf, 'Color', 'w')
        export_fig main_fmri_3_R2_p_ARs.eps
        
        hf = figure;
        hf.Position(3) = 640;
        boxplot(whiteness_p_rec_2D(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', 1:numel(plot_ind), 'XTickLabelRotation', 90, 'xticklabel', labels(plot_ind), ...
            'fontsize', 25, 'yscale', 'log', 'TickLabelInterpreter', 'latex')
        xlims = [0.5 n_plot+0.5];
        hold on
        plot(xlims, [0.05 0.05], 'k--')
        ylabel('Whiteness p-value', 'Interpreter', 'latex')
        grid on
        xlim(xlims)
        ylim([1e-10 1])
        hf.Color = 'w';
        export_fig main_fmri_3_p_ARs.eps
        
        figure
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = ranksum(whiteness_p_rec_2D(:, plot_ind(i)), whiteness_p_rec_2D(:, plot_ind(j)), ...
                    'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind), [], 20)
        set(gcf, 'Color', 'w')
        export_fig main_fmri_3_p_p_ARs.eps
        
        hf = figure;
        hf.Position(3) = 640;
        boxplot(runtime_rec(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', 1:numel(plot_ind), 'XTickLabelRotation', 90, 'xticklabel', labels(plot_ind), ...
            'ytick', 10.^(-1:4), 'fontsize', 25, 'yscale', 'log', 'TickLabelInterpreter', 'latex')
        ylabel('Run Time', 'Interpreter', 'latex')
        xlim([0.5 n_plot+0.5])
        ylim([1e-1 1e4])
        grid on
        hf.Color = 'w';
        export_fig main_fmri_3_time_ARs.eps
        
        figure
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = ranksum(runtime_rec(:, plot_ind(i)), runtime_rec(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind), [], 20)
        set(gcf, 'Color', 'w')
        export_fig main_fmri_3_time_p_ARs.eps
        
        % --- R^2 distribution figures
        hf = figure;                                                        % Violin plots of the distribution of R^2 of the best model over each resting state network.
        hf.Position(3) = hf.Position(3) + 100;
        T = readtable('Schaefer2018_100Parcels_7Networks_order.txt');
        parcel_labels = T.Var2;
        network_labels = {'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'};
        network_labels_ext = {'\parbox{4.3em}{\centering Visual}', ...
            '\parbox{4.3em}{\centering Somato- \\[-2pt] motor}', ...
            '\parbox{4.3em}{\centering Dorsal \\[-2pt] Attention}', ...
            '\parbox{4.3em}{\centering Ventral \\[-2pt] Attention}', ...
            '\parbox{4.3em}{\centering Limbic}', ...
            '\parbox{4.3em}{\centering Fronto- \\[-2pt] parietal}', ...
            '\parbox{4.3em}{\centering Default \\[-2pt] Mode}', ...
            '\parbox{4.3em}{\centering Subcortex}'};
        T = readtable('Yeo2011_7Networks_ColorLUT.txt');
        network_colors = num2cell([table2array(T(2:end, 3:5))/255; 0.5*ones(1, 3)], 2);
        network2parcel = nan(100, 7);
        for i_net = 1:7
            network2parcel(:, i_net) = contains(parcel_labels, network_labels{i_net});
        end
        network2parcel = blkdiag(network2parcel, true(16, 1));
        network2parcel(~network2parcel) = nan;
        R2_network = permute(mean(R2_rec(:, 6, :) .* network2parcel, 'omitnan'), [3 2 1]);
        [~, sort_ind] = sort(median(R2_network), 'descend');
        R2_network = R2_network(:, sort_ind);
        network_labels_ext = network_labels_ext(sort_ind);
        network_colors = network_colors(sort_ind);
        distributionPlot(R2_network, 'distWidth', 0.7, 'color', network_colors, 'showMM', 0)
        set(gca, 'xticklabel', network_labels_ext, 'TickLabelInterpreter', 'latex', 'fontsize', 20, ...
            'XTickLabelRotation', 90)
        ylabel('Average $R^2$', 'Interpreter', 'latex')
        grid on
        set(gcf, 'color', 'w')
        export_fig main_fmri_3_nets.eps
        %
        n_net = numel(network_labels_ext);
        P = nan(n_net);                                                     % The matrix of pairwise p-values for one-sided comparing between the distributions of average R^2 of the best model between resting state networks.
        for i = 1:n_net
            for j = 1:n_net
                P(i, j) = ranksum(R2_network(:, i), R2_network(:, j), 'tail', 'right');
            end
        end
        alpha = 0.05;
        n_pair = nchoosek(n_net, 2);
        min_p = min(P, P');
        min_p_vec = sort(min_p(tril(true(n_net), -1)), 'ascend');
        max_p_ind = find(min_p_vec < (1:n_pair)'/n_pair*alpha, 1, 'last');
        is_significant = P <= min_p_vec(max_p_ind);
        
        hf = figure;                                                        % The cortical distribution of average regional R^2 of the best model (model number 6, which is the VAR-3 model)
        R2_max = mean(R2_rec(1:100, 6, :), 3);
        hf.Position(3) = 700;
        min_color = 1 - (1 - network_colors{1}) * 0.1;
        mid_color = network_colors{1};
        max_color = network_colors{1} * 0.1;
        cmap = interp1([0 0.5 1], [min_color; mid_color; max_color], 0:0.001:1);
        plot_Schaefer100(R2_max, cmap)
        hf.Color = 'w';
        export_fig main_fmri_3_cortex.eps
end
end

%% Auxiliary functions
function write_main_sh                                                      % This function writes the shell script main.sh to the same directory. This is used for running jobs on a cluster. The main.sh could be directly written, but in this way all the code is accessible from MATLAB. Note that this is unnecessary if the jobs are going to be run locally.
s = ["#!/bin/bash";
    "matlab -r ""main_fmri(1.5, '$1', $2); exit"""];
fileID = fopen('main.sh', 'w');
fprintf(fileID, '%s\n', s);
end

function plot_p_cmp(p, labels, correction, fontsize, log10p_range)          % Function for plotting a lower triangular matrix of p-values, with hot colors corresponding to entires (i, j), i > j, such that p(i, j) < p_thr, cold colors to entires (i, j), i > j, such that p(j, i) < p_thr, and gray hatches if neither of p(i, j) or p(j, i) is less than p_thr. p_thr is the corrected of 0.05 using either the Bonferroni or FDR correction. 
if nargin < 3 || isempty(correction)
    correction = 'FDR';
end
if nargin < 4 || isempty(fontsize)
    fontsize = 15;
end
if nargin < 5 || isempty(log10p_range)
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
    'ticklabelinterpreter', 'latex', 'fontsize', fontsize, 'YDir', 'reverse')
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