function main_fmri(section, resolution, k, varargin)
%MAIN_FMRI The wrapper code for reproducing the fMRI analyses and graphics
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
%   resolution can take any of 3 values:
%       'coarse': 116 brain parcellations (Schaefer 100x7 + Melbourne Scale
%       I)
%       'fine': 450 parcellations (Schaefer 400x17 + Melbourne Scale III)
%       'vertex': all vertices/voxels in one of the parcels of the 'fine'
%       resolution. If this resolution is chosen, the last element of
%       varargin must be the index of the selected parcel (either of 41,
%       101, or 413 in the above study)
% 
%       Note that this function does not perform any preprocessing or
%       parcellation. All the data are assumed to be preprocessed according
%       to the desired resolution and available in the appropriate
%       directories (see under section == 'run' for the default
%       directories).
% 
%   Additional input arguments:
% 
%   if section == 'submit': varargin{1} = maximum number of jobs to be
%   simultaneously submitted to the cluster, varargin{2} = parcel index (if
%   resolution == 'vertex')
% 
%   if section == 'run': varargin{1} = HCP subject id, varargin{2} =
%   cross-validation index (1-8), varargin{3} = parcel index (if resolution
%   == 'vertex')
% 
%   if section == 'gather' or 'plot': varargin{1} = parcel index (if
%   resolution == 'vertex')
% 
%   IMPORTANT NOTE: If running on cluster, this code assumes that this
%   function (main_fmri) is already pre-compiled (using mcc -m main_fmri.m)
%   to prevent license over-use. That would automatically create the
%   run_main_fmri.sh shell code used below, and has to be repeated whenever
%   this function or any of the functions in its call chain is changed.
%   Note that for precompilation, all the required files (other packages,
%   etc.) must be in the same folder as this function (not just on MATLAB
%   path).
% 
%   Copyright (C) 2021, Erfan Nozari
%   All rights reserved.

run_on_cluster = 1;                                                         % Switch to determine if this code is run on a SGE-managed cluster, where different subject data can be processed in parallel, or not, where all subjects will be run in sequence.

if ~run_on_cluster
    % Adding the parent directory and its sub-directories to the path
    full_filename = mfilename('fullpath');
    slash_loc = strfind(full_filename, '/');
    addpath(genpath(full_filename(1:slash_loc(end))))
end

load subjs.mat subjs                                                        % subjs is a cell array of HCP subjects used for the study.
n_subj = numel(subjs);

rng(0)

subj_sel_ratio = 1;                                                       % The ratio of subjects used for all the ensuing analyses.
sel_ind = sort(randperm(n_subj, round(n_subj * subj_sel_ratio)), 'ascend');
subjs = subjs(sel_ind);
n_subj = numel(subjs);

switch resolution
    case 'coarse'
        n_cv = 8;                                                           % Number of cross-validation folds.
        i_cv_vals = repmat(1:n_cv, n_subj, 1);                              % Index of the cross validations (out of the possible n_cv) that should be done for each subject.
        mem = 16;                                                           % Memory (in GB) that should be requested for each cluster job.
    case {'fine', 'vertex'}
        n_cv = 1;
        i_cv_vals = randi(8, n_subj, 1);
        mem = 64;
end

switch section
    case 'submit'
        if ~exist(['main_data_' resolution], 'dir')
            mkdir(['main_data_' resolution])                                % Directory to store the created data files. main_fmri('gather', ...) will later look inside this folder.
        end
        if ~exist('main_data_dnn', 'dir')                                   % Directory to keep track of which jobs have run to the point of running the MLP method. This is important since the MLP trainer occationally hangs indefinitely, and the job needs to be killed and restarted (done automatically by the code below).
            mkdir('main_data_dnn')
        end 
        
        if run_on_cluster
            if numel(varargin) >= 1
                max_jobs = varargin{1};                                     % The maximum number of jobs that should be submitted to the cluster at once.
            else
                max_jobs = inf;
            end
            if isequal(resolution, 'vertex')
                parcel = varargin{2};                                       % The index of the parcel all of whose vertices/voxels should be used for system id.
            else
                parcel = nan;
            end
            write_main_sh(mem);                                             % Function to write the shell code main.sh for submitting SGE jobs.
            system('chmod +x main.sh');                                     % Making the shell code generated by the line above executable.
            
            % Record-keeping for what jobs are still running on the cluster
            % and which have completed.
            if exist('job_id_rec.mat', 'file')
                load job_id_rec.mat job_id_rec                              % job_id_rec is a record of all the jobs that have been actively running on the cluster since the last call to this function.
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
            
            % Checking all the active jobs for potential hung status of the
            % DNN (MLP) model, in which case the job is deleted and will be
            % automatically resubmitted the next time main_fmri('submit',
            % ...) is called.
            job_id_del = [];                                                % List of jobs to be deleted, because the DNN trainer has not returned for at least 2 minutes. Adjust this time for your own dataset.
            for job_id = job_id_active'
                job_id_rec_eq_job_id = [job_id_rec{:, 2}] == job_id;
                if nnz(job_id_rec_eq_job_id) == 1
                    subj_i_cv = job_id_rec{job_id_rec_eq_job_id, 1};
                    filename = ['main_data_dnn_' resolution '/' subj_i_cv '_dnn.mat'];
                    if exist(filename, 'file')
                        load(filename, 'dnn_dt')
                        if datetime - dnn_dt > minutes(10)
                            job_id_del(end+1) = job_id;
                            delete(filename)
                        end
                    end
                else
                    warning(['job_id ' num2str(job_id) ' has ' num2str(nnz(job_id_rec_eq_job_id)) ' matches in job_id_rec.'])
                end
            end
            if ~isempty(job_id_del)
                system(['qdel ' num2str(job_id_del)]);                      % Killing the jobs in job_id_del from the cluster
            end
            job_id_active = setdiff(job_id_active, job_id_del);
            
            job_id_rec_isactive = ismember([job_id_rec{:, 2}], job_id_active);
            job_id_rec = job_id_rec(job_id_rec_isactive, :);                % Updating job_id_rec by removing the jobs that have either completed since the last call or killed above
            
            % Moving the output and error logs of all completed/killed jobs
            % to a local trash folder for future inspection
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
            
            n_jobs = 0;                                                     % Number of submitted jobs in this round, needed to ensure the number of all active jobs does not exceed max_jobs.
            for i_subj = 1:n_subj
                subj = subjs{i_subj};
                for i_cv = i_cv_vals(i_subj, :)
                    switch resolution
                        case {'coarse', 'fine'}
                            filename = ['main_data_' resolution '/' subj '_' num2str(i_cv) '.mat']; % The filename where the results of this subject-cross validation should have been saved if it has been run and completed already.
                        case 'vertex'
                            filename = ['main_data_' resolution '/' subj '_' num2str(i_cv) '_' num2str(parcel) '.mat']; % The filename where the results of this subject-cross validation should have been saved if it has been run and completed already.
                    end
                    if ~exist(filename, 'file') ... &&
                            && ~ismember([subj '_' num2str(i_cv)], job_id_rec(:, 1)) ... &&
                            && n_jobs < max_jobs - numel(job_id_active)     % First it checks whether that subject-cross validation has been completed before. This is useful if max_jobs is not infinity, so main_fmri('submit', ...) needs to be called multiple times until all subject-cross validations are completed. Then it checks if that job is currently being run on a node, and finally if there is room for submitting new jobs.
                        system('rm -f /cbica/home/nozarie/.matlab/R2018a/toolbox_cache-9.4.0-3284594471-glnxa64.xml');
                        [~, cmdout] = system(['qsub ./main.sh ' resolution ' ' num2str(k) ' ' subj ' ' num2str(i_cv) ' ' num2str(parcel)]);
                        job_id_rec(end+1, :) = {[subj '_' num2str(i_cv)], ...
                            str2double(cmdout(strfind(cmdout, 'Your job ')+(9:15)))}; % Keeping record of the just-submitted job
                        n_jobs = n_jobs + 1;
                    end
                end
            end
            
            save job_id_rec.mat job_id_rec
        else                                                                % If run_on_cluster == 0, the jobs are serially run on the current machine until complete.
            for i_subj = 1:n_subj
                subj = subjs{i_subj};
                for i_cv = i_cv_vals(i_subj, :)
                    main_fmri('run', resolution, k, subj, i_cv, parcel)
                end
            end
        end
        
    case 'run'
        if ischar(k)
            k = str2double(k);
        end
        subj = varargin{1};
        i_cv = varargin{2};
        if ischar(i_cv)
            i_cv = str2double(i_cv);
        end
        parcel = varargin{3};
        if ischar(parcel)
            parcel = str2double(parcel);
        end
        scans = {'1_LR', '1_RL', '2_LR', '2_RL'};                           % The set of IDs for the four resting state scans that each subject has under the HCP protocol.
        n_scan = numel(scans);
        Y = cell(1, n_scan);                                                % A cell array of data segments used for system identification. Each element of Y is one scan, with channels along the first dimension and time along the second dimension.
        Y_pw = cell(1, n_scan);
        for i_scan = 1:n_scan
            scan = scans{i_scan};
            switch resolution                                               % Reading the pre-processed and parcellated time series for each scan. Change this to your data directory and filename if different.
                case 'coarse'
                    Y{i_scan} = readNPY(['HCP/coarse/yeo_100_' subj '_REST' scan '.npy']);
                case 'fine'
                    load(['HCP/fine/' subj '_' scan '.mat'], 'V');
                    Y{i_scan} = double(V);
                case 'vertex'
                    load(['HCP/vertex/' subj '_' scan '_' num2str(parcel) '.mat'], 'V')
                    Y{i_scan} = double(V);
            end
            Y_pw{i_scan} = Y{i_scan}(:, end/4+1:end/2);                     % Using only the second quarter of each scan for pairwise estimates to reduce computational complexity. Significantly less data is needed as well anyways in 2 dimensions vs. 116!
        end
        TR = 0.72;                                                          % The sampling time of HCP. Change to the sampling time of your dataset if different.
        n_cv = 8;                                                           % Overwriting n_cv only to be used in the next line.
        test_range = [(i_cv-1)/n_cv, i_cv/n_cv];                            % A sub-interval of [0, 1] indicating the portion of the data that is used for test (cross-validation). The rest of the data is used for training.
        MMSE_memory = -mem;                                                 % Memory code for the MMSE method. See MMSE_est.m for details.
        [model_summary, R2, R2_pw, runtime, whiteness, whiteness_pw, model, Y_hat] = ...
            all_methods_fmri(Y, Y_pw, TR, k, test_range, MMSE_memory, [subj '_' num2str(i_cv)]); % Calling the routine that runs all system id methods.

        switch resolution
            case {'coarse', 'fine'}
                filename = ['main_data_' resolution '/' subj '_' num2str(i_cv) '.mat'];
            case 'vertex'
                filename = ['main_data_' resolution '/' subj '_' num2str(i_cv) '_' num2str(parcel) '.mat'];
        end
        save(filename, 'model_summary','R2', 'R2_pw', 'runtime', 'whiteness', 'whiteness_pw', ...
            'Y_hat', 'model', '-v7.3');                               % Saving the data to be used in the subsequent 'gather' section.
    
    case 'gather'
        if isequal(resolution, 'vertex')
            parcel = varargin{1};
        end
        
        R2_rec = cell(1, 1, n_subj);                                        % Cell array for collecting all R^2 vectors from methods.
        R2_pw_rec = cell(1, 1, 1, n_subj);                                  % Same, but for pairwise methods (pairwise linear and pairwise MMSE).
        runtime_rec = cell(n_subj, 1);                                      % Same, but for the time that each method takes to run.
        whiteness_rec = cell(n_subj, 1, n_cv);                              % Similar, but for the whiteness structure of each method.
        whiteness_pw_rec = cell(n_subj, 1, n_cv);                           % Same, but for pairwise methods.
        subj_cv_done = false(n_subj, n_cv);                                 % Array of flags that indicates whether each subject-cross validation has been completed and its data is avialable
        
        n = 116 * isequal(resolution, 'coarse') ... +
            + 450 * isequal(resolution, 'fine') ... +
            + 154 * (isequal(resolution, 'vertex') && parcel == 41) ... +
            + 152 * (isequal(resolution, 'vertex') && parcel == 101) ... +
            + 177 * (isequal(resolution, 'vertex') && parcel == 413);       % Number of channels (parcels or vertices/voxels)
        n_method = 16;                                                      % Total number of system id methods used (not including pairwise methods)
        n_method_pw = 2;                                                    % Number of pairwise methods run. This and the above 2 lines must be consistent with all_methods_fmri.m.    

        for i_subj = 1:n_subj
            subj = subjs{i_subj};
            R2_subj_rec = nan(n, n_method, n_cv);                           % All R2's for all cross validations of that subject. Simiarly for the next two lines.
            R2_pw_subj_rec = nan(n, n, n_method_pw, n_cv);
            runtime_subj_rec = nan(n_cv, n_method+n_method_pw);
            for i_cv = i_cv_vals(i_subj, :)
                switch resolution
                    case {'coarse', 'fine'}
                        filename = ['main_data_' resolution '/' subj '_' num2str(i_cv) '.mat']; % The filename where the results of this subject-cross validation should have been saved if it has been run and completed already.
                    case 'vertex'
                        filename = ['main_data_' resolution '/' subj '_' num2str(i_cv) '_' num2str(parcel) '.mat']; % The filename where the results of this subject-cross validation should have been saved if it has been run and completed already.
                end
                if exist(filename, 'file')
                    load(filename, 'R2', 'R2_pw', 'runtime', 'whiteness', 'whiteness_pw')
                    if any(strcmp(resolution, {'fine', 'vertex'}))
                        i_cv = 1;                                           % Overwriting i_cv only for the next few lines.
                    end
                    R2_subj_rec(:, :, i_cv) = R2;
                    R2_pw_subj_rec(:, :, :, i_cv) = R2_pw;
                    runtime_subj_rec(i_cv, :) = runtime;
                    whiteness_rec{i_subj, 1, i_cv} = whiteness;             % The variable whiteness is a struct, and is therefore treated differently from R2 and runtime. It is not averaged here, but rather later when plotting.
                    whiteness_pw_rec{i_subj, 1, i_cv} = whiteness_pw;       % Same for whiteness_pw.
                    subj_cv_done(i_subj, i_cv) = true;
                end
            end
            R2_rec{i_subj} = mean(R2_subj_rec, 3, 'omitnan');               % Averaging over cross-validation folds, same for the subsequent four lines.
            R2_pw_rec{i_subj} = mean(R2_pw_subj_rec, 4, 'omitnan');
            runtime_rec{i_subj} = mean(runtime_subj_rec, 1, 'omitnan');
        end

        R2_rec = cell2mat(R2_rec);                                          % Transforming from cell array to numerical array that takes less memory. Same below. This is n x n_method x n_subj.
        R2_pw_rec = cell2mat(R2_pw_rec);                                    % n x n_method_pw x n_subj
        runtime_rec = cell2mat(runtime_rec);                                % n_subj x (n_method + n_method_pw)
        whiteness_rec = cell2mat(whiteness_rec(all(subj_cv_done, 2), :, :)); % n_subj x n_method x n_cv (struct array)
        whiteness_pw_rec = cell2mat(whiteness_pw_rec(all(subj_cv_done, 2), :, :)); % n_subj x n_method_pw x n_cv (struct array)
        
        switch resolution
            case {'coarse', 'fine'}
                filename = ['main_data_' resolution '.mat'];
            case 'vertex'
                filename = ['main_data_' resolution '_' num2str(parcel) '.mat'];
        end
        save(filename, 'R2_rec', 'R2_pw_rec', 'runtime_rec', 'whiteness_rec', 'whiteness_pw_rec', 'subjs', 'n_cv', '-v7.3')
    
    case 'plot'
        switch resolution
            case {'coarse', 'fine'}
                filename = ['main_data_' resolution '.mat'];
            case 'vertex'
                parcel = varargin{1};
                filename = ['main_data_' resolution '_' num2str(parcel) '.mat'];
        end
        load(filename, 'R2_rec', 'R2_pw_rec', 'runtime_rec', 'whiteness_rec', 'whiteness_pw_rec')
        [n, n_method, n_subj] = size(R2_rec);
        R2_rec_2D = reshape(permute(R2_rec, [2 1 3]), n_method, n * n_subj)'; % Flattening R2_rec so that each column contains combined data for all subjects and all brain regions for any given method. Same below.
        n_method_pw = size(R2_pw_rec, 3);
        R2_pw_rec_2D = cell2mat(arrayfun(@(i_method)reshape(R2_pw_rec(:, :, i_method, :), [], 1), ...
            1:n_method_pw, 'UniformOutput', 0));                            % Same as R2_rec_2D
        whiteness_stat_rec_2D = mean(reshape([whiteness_rec.stat], [], n_method, n_cv), 3); % Average across cross validations of the whiteness statistic
        whiteness_sig_thr_rec_2D = mean(reshape([whiteness_rec.sig_thr], [], n_method, n_cv), 3); % Same for whiteness threshold
        whiteness_rel_stat_rec_2D = whiteness_stat_rec_2D ./ whiteness_sig_thr_rec_2D; % The ratio between average whiteness statistic and average whiteness ratio
        whiteness_pw_stat_rec_2D = reshape(mean(cell2mat(reshape({whiteness_pw_rec.stat}, ...
            1, 1, [], n_method_pw, n_cv)), 5), [], n_method_pw);
        whiteness_pw_rel_stat_rec_2D = whiteness_pw_stat_rec_2D / chi2inv(0.95, 20);
        
        plot_ind = setdiff(1:n_method, [4 5 7]);                            % List of methods (not pairwise methods) to including in all plotting below. Uninteresting methods are taken out.
        n_plot = numel(plot_ind);
        plot_loc_pw = numel(plot_ind)+(2:3);                                % To be able to plot boxplots of brain-wise and pairwise methods side by side in a single axis
        n_lin_method = 8;
        n_nonlin_method = 7;
        colors = [0.3*ones(1, 3); repmat(matlab_green, n_lin_method, 1); ...
            repmat(matlab_yellow, n_nonlin_method, 1)];                     % Colors of boxplots for each method
        colors_pw = [matlab_green; matlab_yellow];
        labels = {'Zero';
            'Linear\\[-2pt](dense)';
            'Linear\\[-2pt](sparse)';
            'VAR-2\\[-2pt](sparse)';
            'AR-2\\[-2pt](sparse)';
            'VAR-3\\[-2pt](sparse)';
            'AR-3\\[-2pt](sparse)';
            'Linear\\[-2pt]w/ HRF';
            'Subspace';
            'NMM';
            'NMM\\[-2pt]w/ HRF';
            'Manifold';
            'DNN\\[-2pt](MLP)';
            'DNN\\[-2pt](CNN)';
            'LSTM\\[-2pt](IIR)';
            'LSTM\\[-2pt](FIR)'};                                           % Labels abbreviating each method
        labels = cellfun(@(label)['\parbox{4.5em}{\centering ' label '}'], labels, 'UniformOutput', 0); % Small modification for better latex rendering
        labels_pw = {'Linear\\[-2pt](pairwise)'; 
            'MMSE\\[-2pt](pairwise)'};
        labels_pw = cellfun(@(label)['\parbox{4.5em}{\centering ' label '}'], labels_pw, 'UniformOutput', 0);
        
        % --- Methods figures
        hf = figure;                                                        % The boxplot containing the R^2 comparisons
        hf.Position(3) = 640;
        boxplot(R2_rec_2D(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        hold on
        boxplot(R2_pw_rec_2D, 'Positions', plot_loc_pw, 'Whisker', inf, 'Colors', colors_pw, 'Widths', 0.5)
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', [1:n_plot plot_loc_pw], 'XTickLabelRotation', 90, ...
            'xticklabel', [labels(plot_ind); labels_pw], 'fontsize', 15, 'TickLabelInterpreter', 'latex')
        if isequal(resolution, 'coarse')
            ylims = get(gca, 'ylim');
            ylim([ylims(1) 1])
        end
        ylabel('$R^2$', 'Interpreter', 'latex')
        xlim([0.5 plot_loc_pw(end)+0.5])
        grid on
        switch resolution
           case 'coarse'
               exportgraphics(gcf, 'main_3_R2.eps')
           case 'fine'
               exportgraphics(gcf, ['main_3_R2_' resolution '.eps'])
           case 'vertex'
               exportgraphics(gcf, ['main_3_R2_' resolution '_' num2str(parcel) '.eps'])
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
        switch resolution
            case 'coarse'
                exportgraphics(gcf, 'main_3_R2_p.eps')
            case 'fine'
                exportgraphics(gcf, ['main_3_R2_p_' resolution '.eps'])
            case 'vertex'
                exportgraphics(gcf, ['main_3_R2_p_' resolution '_' num2str(parcel) '.eps'])
        end
        
        hf = figure;                                                        % Same as above but for pairwise methods (together with the zero method, which is taken from the R2_rec)
        hf.Position(3:4) = hf.Position(3:4) - 270;
        p = nan(n_method_pw+1);
        R2_pw_rec_2D = [{R2_rec_2D(:, 1)} fliplr(num2cell(R2_pw_rec_2D, 1))];
        for i = 1:n_method_pw+1
            for j = 1:n_method_pw+1
                if any([i j] == 1)
                    p(i, j) = ranksum(R2_pw_rec_2D{i}, R2_pw_rec_2D{j}, 'tail', 'right');
                else
                    p(i, j) = signrank(R2_pw_rec_2D{i}, R2_pw_rec_2D{j}, 'tail', 'right');
                end
            end
        end
        log10p_range = [-5 0] + [-35 0] * isequal(resolution, 'coarse');
        plot_p_cmp(p, [labels(1); flipud(labels_pw)], [], [], log10p_range)
        hf.Children(1).Ticks(2:end-1) = [];
        hf.Children(1).TickLabels(2:end-1) = [];
        hf.Color = 'w';
        switch resolution
            case 'coarse'
                exportgraphics(gcf, 'main_3_R2_pw_p.eps')
            case 'fine'
                exportgraphics(gcf, ['main_3_R2_pw_p_' resolution '.eps'])
            case 'vertex'
                exportgraphics(gcf, ['main_3_R2_pw_p_' resolution '_' num2str(parcel) '.eps'])
        end

        hf = figure;                                                        % Same as main_3_R2.eps but for p values of chi-squared test of whiteness
        hf.Position(3) = 640;
        boxplot(whiteness_rel_stat_rec_2D(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        hold on
        boxplot(whiteness_pw_rel_stat_rec_2D, 'Whisker', inf, 'Positions', plot_loc_pw, 'Colors', colors_pw, ...
            'Widths', 0.5)
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', [1:numel(plot_ind) plot_loc_pw], 'XTickLabelRotation', 90, ...
            'xticklabel', [labels(plot_ind); labels_pw], 'fontsize', 15, 'yscale', 'log', ...
            'TickLabelInterpreter', 'latex')
        xlims = [0.5 plot_loc_pw(end)+0.5];
        ylabel('Whiteness Statistic ($Q / Q_{\rm thr}$)', 'Interpreter', 'latex')
        grid on
        xlim(xlims)
        ylim([0.99 2])
        switch resolution
           case 'coarse'
               exportgraphics(gcf, 'main_3_p.eps')
           case 'fine'
               exportgraphics(gcf, ['main_3_p_' resolution '.eps'])
           case 'vertex'
               exportgraphics(gcf, ['main_3_p_' resolution '_' num2str(parcel) '.eps'])
        end
        
        figure                                                              % Same as main_3_R2_p.eps but for p values of chi-squared test of whiteness
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = signrank(whiteness_rel_stat_rec_2D(:, plot_ind(i)), ...
                    whiteness_rel_stat_rec_2D(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind), [], [], log10p_range)
        set(gcf, 'Color', 'w')
        switch resolution
           case 'coarse'
               exportgraphics(gcf, 'main_3_p_p.eps')
           case 'fine'
               exportgraphics(gcf, ['main_3_p_p_' resolution '.eps'])
           case 'vertex'
               exportgraphics(gcf, ['main_3_p_p_' resolution '_' num2str(parcel) '.eps'])
        end
        
        hf = figure;                                                        % Same as main_3_R2_pw_p.eps but for p values of chi-squared test of whiteness
        hf.Position(3:4) = hf.Position(3:4) - 270;
        p = nan(n_method_pw+1);
        whiteness_pw_rel_stat_rec_2D = [{whiteness_rel_stat_rec_2D(:, 1)} ...
            fliplr(num2cell(whiteness_pw_rel_stat_rec_2D, 1))];
        for i = 1:n_method_pw+1
            for j = 1:n_method_pw+1
                if any([i j] == 1)
                    p(i, j) = ranksum(whiteness_pw_rel_stat_rec_2D{i}, ...
                        whiteness_pw_rel_stat_rec_2D{j}, 'tail', 'right');
                else
                    p(i, j) = signrank(whiteness_pw_rel_stat_rec_2D{i}, ...
                        whiteness_pw_rel_stat_rec_2D{j}, 'tail', 'right');
                end
            end
        end
        plot_p_cmp(p, [labels(1); flipud(labels_pw)], [], [], log10p_range)
        hf.Children(1).Ticks(2:end-1) = [];
        hf.Children(1).TickLabels(2:end-1) = [];
        hf.Color = 'w';
        switch resolution
           case 'coarse'
               exportgraphics(gcf, 'main_3_p_pw_p.eps')
           case 'fine'
               exportgraphics(gcf, ['main_3_p_pw_p_' resolution '.eps'])
           case 'vertex'
               exportgraphics(gcf, ['main_3_p_pw_p_' resolution '_' num2str(parcel) '.eps'])
        end
        
        hf = figure;                                                        % Same as main_3_R2.eps but for run times
        hf.Position(3) = 640;
        boxplot(runtime_rec(:, plot_ind), 'Whisker', inf, 'Colors', colors(plot_ind, :))
        hold on
        boxplot(runtime_rec(:, end-1:end), 'Whisker', inf, 'Positions', plot_loc_pw, 'Colors', colors_pw, ...
            'Widths', 0.5)
        set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
        set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
        set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
        set(gca, 'xtick', [1:numel(plot_ind) plot_loc_pw], 'XTickLabelRotation', 90, ...
            'xticklabel', [labels(plot_ind); labels_pw], 'ytick', 10.^(-1:4), 'fontsize', 15, 'yscale', 'log', ...
            'TickLabelInterpreter', 'latex')
        ylabel('Run Time', 'Interpreter', 'latex')
        xlim([0.5 plot_loc_pw(end)+0.5])
        if isequal(resolution, 'coarse')
            ylim([1e-1 1e4])
        end
        grid on
        hf.Color = 'w';
        switch resolution
           case 'coarse'
               exportgraphics(gcf, 'main_3_time.eps')
           case 'fine'
               exportgraphics(gcf, ['main_3_time_' resolution '.eps'])
           case 'vertex'
               exportgraphics(gcf, ['main_3_time_' resolution '_' num2str(parcel) '.eps'])
        end
        
        figure                                                              % Same as main_3_R2_p.eps but for run times
        p = nan(n_plot);
        for i = 1:n_plot
            for j = 1:n_plot
                p(i, j) = signrank(runtime_rec(:, plot_ind(i)), runtime_rec(:, plot_ind(j)), 'tail', 'right');
            end
        end
        plot_p_cmp(p, labels(plot_ind), [], [], log10p_range)
        set(gcf, 'Color', 'w')
        switch resolution
           case 'coarse'
               exportgraphics(gcf, 'main_3_time_p.eps')
           case 'fine'
               exportgraphics(gcf, ['main_3_time_p_' resolution '.eps'])
           case 'vertex'
               exportgraphics(gcf, ['main_3_time_p_' resolution '_' num2str(parcel) '.eps'])
        end
        
        hf = figure;                                                        % Same as main_3_R2_pw_p.eps but for run times
        hf.Position(3:4) = hf.Position(3:4) - 270;
        p = nan(n_method_pw+1);
        runtime_pw_rec = [{runtime_rec(:, 1)} fliplr(num2cell(runtime_rec(:, end-1:end), 1))];
        for i = 1:n_method_pw+1
            for j = 1:n_method_pw+1
                if any([i j] == 1)
                    p(i, j) = ranksum(runtime_pw_rec{i}, runtime_pw_rec{j}, 'tail', 'right');
                else
                    p(i, j) = signrank(runtime_pw_rec{i}, runtime_pw_rec{j}, 'tail', 'right');
                end
            end
        end
        plot_p_cmp(p, [labels(1); flipud(labels_pw)], [], [], log10p_range)
        hf.Children(1).Ticks(2:end-1) = [];
        hf.Children(1).TickLabels(2:end-1) = [];
        hf.Color = 'w';
        switch resolution
           case 'coarse'
               exportgraphics(gcf, 'main_3_time_pw_p.eps')
           case 'fine'
               exportgraphics(gcf, ['main_3_time_pw_p_' resolution '.eps'])
           case 'vertex'
               exportgraphics(gcf, ['main_3_time_pw_p_' resolution '_' num2str(parcel) '.eps'])
        end
        
        if isequal(resolution, 'coarse')
            % --- Linear methods figures (for SI). Figures parallel those above
            plot_ind_lin = 1:7;                                             % List of linear methods (excluding lienar pairwise) to including in the following plots.
            n_plot_lin = numel(plot_ind_lin);

            hf = figure;
            hf.Position(3) = 640;
            boxplot(R2_rec_2D(:, plot_ind_lin), 'Whisker', inf, 'Colors', colors(plot_ind_lin, :))
            set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
            set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
            set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
            set(gca, 'xtick', 1:n_plot_lin, 'XTickLabelRotation', 90, ...
                'xticklabel', labels(plot_ind_lin), 'fontsize', 20, 'TickLabelInterpreter', 'latex')
            ylims = get(gca, 'ylim');
            ylim([ylims(1) 1])
            ylabel('$R^2$', 'Interpreter', 'latex')
            xlim([0.5 n_plot_lin+0.5])
            grid on
            exportgraphics(gcf, 'main_3_R2_ARs.eps')

            figure
            p = nan(n_plot_lin);
            for i = 1:n_plot_lin
                for j = 1:n_plot_lin
                    p(i, j) = signrank(R2_rec_2D(:, plot_ind_lin(i)), R2_rec_2D(:, plot_ind_lin(j)), 'tail', 'right');
                end
            end
            plot_p_cmp(p, labels(plot_ind_lin), [], 20)
            set(gcf, 'Color', 'w')
            exportgraphics(gcf, 'main_3_R2_p_ARs.eps')

            hf = figure;
            hf.Position(3) = 640;
            boxplot(whiteness_rel_stat_rec_2D(:, plot_ind_lin), 'Whisker', inf, 'Colors', colors(plot_ind_lin, :))
            set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
            set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
            set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
            set(gca, 'xtick', 1:numel(plot_ind_lin), 'XTickLabelRotation', 90, 'xticklabel', labels(plot_ind_lin), ...
                'fontsize', 20, 'yscale', 'log', 'TickLabelInterpreter', 'latex')
            xlims = [0.5 n_plot_lin+0.5];
            hold on
            plot(xlims, [0.05 0.05], 'k--')
            ylabel('Whiteness Statistic ($Q / Q_{\rm thr}$)', 'Interpreter', 'latex')
            grid on
            xlim(xlims)
            ylim([1 1.5])
            hf.Color = 'w';
            exportgraphics(gcf, 'main_3_p_ARs.eps')

            figure
            p = nan(n_plot_lin);
            for i = 1:n_plot_lin
                for j = 1:n_plot_lin
                    p(i, j) = signrank(whiteness_rel_stat_rec_2D(:, plot_ind_lin(i)), ...
                        whiteness_rel_stat_rec_2D(:, plot_ind_lin(j)), 'tail', 'right');
                end
            end
            plot_p_cmp(p, labels(plot_ind_lin), [], 20)
            set(gcf, 'Color', 'w')
            exportgraphics(gcf, 'main_3_p_p_ARs.eps')

            hf = figure;
            hf.Position(3) = 640;
            boxplot(runtime_rec(:, plot_ind_lin), 'Whisker', inf, 'Colors', colors(plot_ind_lin, :))
            set(findobj(gcf, 'LineStyle', '-'), 'LineWidth', 3)
            set(findobj(gcf, 'LineStyle', '--'), 'LineWidth', 2)
            set(findobj(gcf, 'LineStyle', '--'), 'LineStyle', '-')
            set(gca, 'xtick', 1:numel(plot_ind_lin), 'XTickLabelRotation', 90, 'xticklabel', labels(plot_ind_lin), ...
                'ytick', 10.^(-1:4), 'fontsize', 20, 'yscale', 'log', 'TickLabelInterpreter', 'latex')
            ylabel('Run Time', 'Interpreter', 'latex')
            xlim([0.5 n_plot_lin+0.5])
            ylim([1e-1 1e4])
            grid on
            hf.Color = 'w';
            exportgraphics(gcf, 'main_3_time_ARs.eps')

            figure
            p = nan(n_plot_lin);
            for i = 1:n_plot_lin
                for j = 1:n_plot_lin
                    p(i, j) = signrank(runtime_rec(:, plot_ind_lin(i)), runtime_rec(:, plot_ind_lin(j)), 'tail', 'right');
                end
            end
            plot_p_cmp(p, labels(plot_ind_lin), [], 20)
            exportgraphics(gcf, 'main_3_time_p_ARs.eps')

            % --- R^2 distribution figures
            hf = figure;                                                    % Violin plots of the distribution of R^2 of the best model over each resting state network.
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
            exportgraphics(gcf, 'main_3_nets.eps')
            %
            n_net = numel(network_labels_ext);
            P = nan(n_net);                                                         % The matrix of pairwise p-values for one-sided comparing between the distributions of average R^2 of the best model between resting state networks.
            for i = 1:n_net
                for j = 1:n_net
                    P(i, j) = signrank(R2_network(:, i), R2_network(:, j), 'tail', 'right');
                end
            end
            alpha = 0.05;
            n_pair = nchoosek(n_net, 2);
            min_p = min(P, P');
            min_p_vec = sort(min_p(tril(true(n_net), -1)), 'ascend');
            max_p_ind = find(min_p_vec < (1:n_pair)'/n_pair*alpha, 1, 'last');
            is_significant = P <= min_p_vec(max_p_ind);

            hf = figure;                                                    % The cortical distribution of average regional R^2 of the best model
            R2_max = mean(R2_rec(1:100, 6, :), 3);
            hf.Position(3) = 700;
            min_color = 1 - (1 - network_colors{1}) * 0.1;
            mid_color = network_colors{1};
            max_color = network_colors{1} * 0.1;
            cmap = interp1([0 0.5 1], [min_color; mid_color; max_color], 0:0.001:1);
            plot_Schaefer100(R2_max, cmap)
            hf.Color = 'w';
            exportgraphics(gcf, 'main_3_cortex.eps')

            for i_plot = 1:n_plot
                hf = figure;                                                % The cortical distribution of average regional R^2 of the model number i_plot
                hf.Position(3) = 700;
                min_color = 1 - (1 - network_colors{1}) * 0.1;
                mid_color = network_colors{1};
                max_color = network_colors{1} * 0.1;
                cmap = interp1([0 0.5 1], [min_color; mid_color; max_color], 0:0.001:1);
                if i_plot == 5
                    clim = [-0.5 max(mean(R2_rec(1:100, plot_ind(5), :), 3))];
                else
                    clim = [];
                end
                plot_Schaefer100(mean(R2_rec(1:100, plot_ind(i_plot), :), 3), cmap, [], [], clim)
                hf.Color = 'w';
                exportgraphics(gcf, ['main_3_cortex_' num2str(i_plot) '.eps'])
            end

            hf = figure;
            imagesc(corrcoef(R2_rec_2D(:, plot_ind)))
            axis equal
            axis([0.5 n_plot+0.5 0.5 n_plot+0.5])
            set(gca, 'fontsize', 12, 'xtick', 1:n_plot, 'XTickLabel', labels(plot_ind), ...
                'ytick', 1:n_plot, 'YTickLabel', labels(plot_ind), ...
                'XTickLabelRotation', 90, 'TickLabelInterpreter', 'latex')
            hc = colorbar;
            hc.Label.String = 'Correlation Coefficient';
            hc.Label.Interpreter = 'latex';
            hc.TickLabelInterpreter = 'latex';
            exportgraphics(gcf, 'main_3_corrcoef.eps')
            
            % Comparison p-values using t-test instead of signrank. Figures
            % parallel those above.
            figure
            p = nan(n_plot);
            for i = 1:n_plot
                for j = 1:n_plot
                    [~, p(i, j)] = ttest(R2_rec_2D(:, plot_ind(i)), R2_rec_2D(:, plot_ind(j)), 'tail', 'right');
                end
            end
            plot_p_cmp(p, labels(plot_ind))
            exportgraphics(gcf, 'main_3_R2_p_ttest.eps')

            hf = figure;
            hf.Position(3:4) = hf.Position(3:4) - 270;
            p = nan(n_method_pw+1);
            for i = 1:n_method_pw+1
                for j = 1:n_method_pw+1
                    if i == 1 || j == 1
                        [~, p(i, j)] = ttest2(R2_pw_rec_2D{i}, R2_pw_rec_2D{j}, 'tail', 'right');
                    else
                        [~, p(i, j)] = ttest(R2_pw_rec_2D{i}, R2_pw_rec_2D{j}, 'tail', 'right');
                    end
                end
            end
            plot_p_cmp(p, [labels(1); flipud(labels_pw)])
            hf.Children(1).Ticks(2:end-1) = [];
            hf.Children(1).TickLabels(2:end-1) = [];
            exportgraphics(gcf, 'main_3_R2_pw_p_ttest.eps')

            figure
            p = nan(n_plot);
            for i = 1:n_plot
                for j = 1:n_plot
                    [~, p(i, j)] = ttest(whiteness_stat_rec_2D(:, plot_ind(i)), ...
                        whiteness_stat_rec_2D(:, plot_ind(j)), 'tail', 'right');
                end
            end
            plot_p_cmp(p, labels(plot_ind))
            exportgraphics(gcf, 'main_3_p_p_ttest.eps')

            hf = figure;
            hf.Position(3:4) = hf.Position(3:4) - 270;
            p = nan(n_method_pw+1);
            for i = 1:n_method_pw+1
                for j = 1:n_method_pw+1
                    if i == 1 || j == 1
                        [~, p(i, j)] = ttest2(whiteness_pw_stat_rec_2D{i}, ...
                            whiteness_pw_stat_rec_2D{j}, 'tail', 'right');
                    else
                        [~, p(i, j)] = ttest(whiteness_pw_stat_rec_2D{i}, ...
                            whiteness_pw_stat_rec_2D{j}, 'tail', 'right');
                    end
                end
            end
            plot_p_cmp(p, [labels(1); flipud(labels_pw)])
            hf.Children(1).Ticks(2:end-1) = [];
            hf.Children(1).TickLabels(2:end-1) = [];
            exportgraphics(gcf, 'main_3_p_pw_p_ttest.eps')

            figure
            p = nan(n_plot);
            for i = 1:n_plot
                for j = 1:n_plot
                    [~, p(i, j)] = ttest(runtime_rec(:, plot_ind(i)), ...
                        runtime_rec(:, plot_ind(j)), 'tail', 'right');
                end
            end
            plot_p_cmp(p, labels(plot_ind))
            exportgraphics(gcf, 'main_3_time_p_ttest.eps')

            hf = figure;
            hf.Position(3:4) = hf.Position(3:4) - 270;
            p = nan(n_method_pw+1);
            for i = 1:n_method_pw+1
                for j = 1:n_method_pw+1
                    if i == 1 || j == 1
                        [~, p(i, j)] = ttest2(runtime_pw_rec{i}, runtime_pw_rec{j}, 'tail', 'right');
                    else
                        [~, p(i, j)] = ttest(runtime_pw_rec{i}, runtime_pw_rec{j}, 'tail', 'right');
                    end
                end
            end
            plot_p_cmp(p, [labels(1); flipud(labels_pw)])
            hf.Children(1).Ticks(2:end-1) = [];
            hf.Children(1).TickLabels(2:end-1) = [];
            exportgraphics(gcf, 'main_3_time_pw_p_ttest.eps')
        end
end
end

%% Auxiliary functions
function write_main_sh(mem)                                                 % This function writes the shell script main.sh to the same directory. This is used for running jobs on a cluster. The main.sh could be directly written, but in this way all the code is accessible from MATLAB. Note that this is unnecessary if the jobs are going to be run locally. The shell code run_main_fmri.sh should already be present in the same directory by running mcc -m main_fmri.m in MATLAB.
s = ["#! /bin/bash";
    "#$ -S /bin/bash";
    "#$ -pe threaded 2";
    "#$ -l h_vmem=" + num2str(mem) + "G";
    "#$ -l s_vmem=" + num2str(mem) + "G";
    "./run_main_fmri.sh $MATLAB_DIR run $1 $2 $3 $4 $5"];
fileID = fopen('main.sh', 'w');
fprintf(fileID, '%s\n', s);
end

function plot_p_cmp(p, labels, correction, fontsize, log10p_range)          % Function for plotting a lower triangular matrix of p-values, with hot colors corresponding to entires (i, j), i > j, such that p(i, j) < p_thr, cold colors to entires (i, j), i > j, such that p(j, i) < p_thr, and gray hatches if neither of p(i, j) or p(j, i) is less than p_thr. p_thr is the corrected of 0.05 using either the Bonferroni or FDR correction.
if nargin < 3 || isempty(correction)
    correction = 'FDR';
end
if nargin < 4 || isempty(fontsize)
    fontsize = 13;
end
if nargin < 5 || isempty(log10p_range)
    log10p_range = [-50 0];
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
        if isempty(max_p_ind)
            max_p_ind = 1;
        end
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