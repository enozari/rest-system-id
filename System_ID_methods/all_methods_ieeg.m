function [model_summary_rec, R2_rec, runtime_rec, whiteness_rec, model_rec, Y_hat_rec, best_model] = ...
    all_methods_ieeg(Y, TR, k, test_range, run_subspace, i_segment, MMSE_memory)
%ALL_METHODS_IEEG The function that calls all the methods of system
% identification for iEEG data discussed in the paper
% E. Nozari et. al., "Is the brain macroscopically linear? A system
% identification of resting state dynamics", 2021.
%
%   Input Arguments
% 
%   Y: a data matrix or cell array of data matrices. Each element of Y (or
%   Y itself) is one resting state scan, with channels along the first
%   dimension and time along the second dimension. This is the only
%   mandatory input.
% 
%   TR: Sampling time. 
% 
%   k: number of multi-step ahead predictions for cross-validation.
% 
%   test_range: a sub-interval of [0, 1] indicating the portion of Y that
%   is used for test (cross-validation). The rest of Y is used for
%   training.
% 
%   i_segment: the index of the time series segment for the current data in
%   Y. This is only used in the subspace method to track whether it has to
%   be run or skipped (as dictated by main_ieeg).
% 
%   MMSE_memory: The memory code used for MMSE estimation. See MMSE_est.m
%   for details. The advised value is minus the GB of available memory.
% 
%   Output Arguments
% 
%   model_summary_rec: a cell array of n_method character vectors
%   describing each method in short. n_method is the number of methods
%   applied.
% 
%   R2_rec: an n x n_method array where n is the number of brain regions.
%   Each element contains the cross-validated R^2 for that region under
%   that method.
% 
%   runtime_rec: an n_method x 1 vector containing the time that each
%   method takes to run.
% 
%   whiteness_rec: an struct object ob size 1 x n_method, containing the
%   statistic (Q) and the randomization-basd significance threshold and
%   p-value of the multivariate whiteness test for each method.
% 
%   model_rec: a cell array of models. Each element of model_rec is a
%   struct with detailed description (functional form and parameters) of
%   the fitted model from any of the model families.
% 
%   Y_hat_rec: an n_method x 1 cell array of predicted time series. Each
%   element of Y_hat_rec is itself a cell array the same size as Y.
%   
%   best_model: a struct containing the model, R^2, etc. for the best
%   model. The best model is the one whose R^2 has the largest median, as
%   compared using a ranksum test.
% 
%   Copyright (C) 2021, Erfan Nozari
%   All rights reserved.

if nargin < 2 || isempty(TR)
    warning('Selecting TR = 0.002 by default. Provide its correct value if different.')
    TR = 0.002;
end
if nargin < 3 || isempty(test_range)
    test_range = [0.8 1];
end
if nargin < 4 || isempty(run_subspace)
    run_subspace = true;
end
if nargin < 5
    MMSE_memory = [];
end

%% Initializing the record-keeping variables
n_method = 13;
exec_order = 1:n_method;                                      % The order in which the methods are run.
model_rec = cell(n_method, 1);
if iscell(Y)
    n = size(Y{1}, 1);
else
    n = size(Y, 1);
end
R2_rec = zeros(n, n_method);
whiteness_rec = struct('p', cell(1, n_method), 'stat', [], 'sig_thr', []);
Y_hat_rec = cell(1, n_method);
model_summary_rec = cell(1, n_method);
runtime_rec = nan(1, n_method);

%% Checking if Parallel Processing Toolbox license is available
status = license('checkout', 'Distrib_Computing_Toolbox');
if status
    use_parallel = 1;
    try
        parpool
    catch
        warning('Initialization of parpool unsuccessful. The runtime of each method involving parfor will include a parpool starting time.')
    end
else
    use_parallel = 0;
end

%% Running all methods
for i_exec = exec_order                                                     % Running different methods one by one, in the order specified by exec_order
    switch i_exec
        case 1
            %% The zero model
            model_summary_rec{i_exec} = 'Zero';
            include_W = 0;                                                  % Whether to include network interconnections (effecitive connectivity), not including self-loops
            n_AR_lags = 0;                                                  % Number of autoregressive lags
            W_mask = [];                                                    % The default sparsity structure of linear_AR (which is irrelevant here since include_W = 0)
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 2
            %% Linear model at the BOLD level with full effective connectivity
            model_summary_rec{i_exec} = 'Linear (dense)';
            include_W = 1;
            n_AR_lags = 1;
            W_mask = 'full';                                                % Dense, potentially all-to-all effective connectivity
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 3
            %% Linear model at the BOLD level with sparse effective connectivity via LASSO regularization
            model_summary_rec{i_exec} = 'Linear (sparse)';
            include_W = 1;
            n_AR_lags = 1;
            W_mask = 1.2;                                                    % LASSO regularization to promote sparsity in effective connectivity
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 4
            %% Linear model at the BOLD level with sparse effective connectivity via LASSO regularization and second order auto-regression
            model_summary_rec{i_exec} = 'AR-100 (sparse)';
            include_W = 1;
            n_AR_lags = 100;
            W_mask = 1.5;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 5
            %% Linear model at the BOLD level with sparse effective connectivity via LASSO regularization and third order auto-regression
            model_summary_rec{i_exec} = 'AR-100 (scalar)';
            include_W = 0;
            n_AR_lags = 102;
            W_mask = [];
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 6
            %% Linear model via subspace identification
            model_summary_rec{i_exec} = 'Subspace';
            if run_subspace
                subspace_t0 = datetime;
                filename = ['main_data_subspace/' num2str(i_segment) '_t0.mat'];
                save(filename, 'subspace_t0', 'n')
                s = 11;
                r = 49;
                n = 436;
                tic
                [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                    linear_subspace(Y, s, r, n, k, test_range);
                runtime_rec(i_exec) = toc;
                delete(filename)
                disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
            else
                R2_rec(:, i_exec) = -inf;
                whiteness_rec(i_exec).p = 0;
                whiteness_rec(i_exec).stat = inf;
                whiteness_rec(i_exec).sig_thr = 0;
                Y_hat_rec{i_exec} = nan(size(Y));
                runtime_rec(i_exec) = inf;
                disp([model_summary_rec{i_exec} ' skipped.'])
            end
            
        case 7
            %% Nonlinear model via MINDy [Singh et al., 2019] applied directly to time series (no HRF/deconvolution)
            model_summary_rec{i_exec} = 'NMM';
            doPreProc = 'n';
            lambda = {0.2 0.2 2 0.5};
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                nonlinear_MINDy(Y, TR, doPreProc, lambda, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 8
            %% Nonlinear model based on locally linear manifold learning
            model_summary_rec{i_exec} = 'Manifold';
            n_AR_lags = 7;
            kernel = 'Gaussian';
            h = 1.2e4;
            scale_h.do_scale = false;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                nonlinear_manifold(Y, n_AR_lags, kernel, h, scale_h, k, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 9
            %% Nonlinear model based on deep neural networks
            model_summary_rec{i_exec} = 'MLP';
            n_AR_lags = 6;
            hidden_width = 26;
            hidden_depth = 4;
            dropout_prob = 0.5;
            if use_parallel
                exe_env = 'auto';                                           % The 'ExecutionEnvironment' option of the neural network toolbox
            else
                exe_env = 'cpu';
            end
            for i = 3:10
                learn_rate = 10^-i
                tic
                [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                    nonlinear_MLP(Y, n_AR_lags, hidden_width, hidden_depth, dropout_prob, ...
                    exe_env, learn_rate, k, test_range);
                runtime_rec(i_exec) = toc;
                if ~any(isnan(R2_rec(:, i_exec)))
                    break
                end
            end
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 10
            %% Nonlinear model based on convolutional neural networks
            model_summary_rec{i_exec} = 'CNN';
            n_AR_lags = 11;
            hidden_depth = 7;
            filter_size = 2;
            n_filter = 13;
            pool_size = 1;
            dilation_factor = [];
            exp_dilation = [];
            dropout_prob = 0.5;
            exe_env = 'cpu';
            for i = 3:6
                learn_rate = 10^-i
                tic
                [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                    nonlinear_CNN(Y, n_AR_lags, hidden_depth, filter_size, n_filter, ...
                    pool_size, dilation_factor, exp_dilation, dropout_prob, ...
                    exe_env, learn_rate, k, test_range);
                runtime_rec(i_exec) = toc;
                if ~any(isnan(R2_rec(:, i_exec)))
                    break
                end
            end
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 11
            %% Nonlinear model based on LSTM neural networks with infinite impulse response
            model_summary_rec{i_exec} = 'LSTM1';
            rec_layer = 'lstm';
            n_AR_lags = nan;
            hidden_width = 1;
            exe_env = 'cpu';
            for i = 2:5
                learn_rate = 10^-i
                tic
                [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                    nonlinear_LSTM(Y, rec_layer, n_AR_lags, hidden_width, exe_env, ...
                    learn_rate, k, test_range);
                runtime_rec(i_exec) = toc;
                if ~any(isnan(R2_rec(:, i_exec)))
                    break
                end
            end
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 12
            %% Nonlinear model based on LSTM neural networks with finite impulse response
            model_summary_rec{i_exec} = 'LSTM2';
            rec_layer = 'lstm';
            n_AR_lags = 7;
            hidden_width = 1;
            exe_env = 'cpu';
            for i = 2:5
                learn_rate = 10^-i
                tic
                [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                    nonlinear_LSTM(Y, rec_layer, n_AR_lags, hidden_width, exe_env, ...
                    learn_rate, k, test_range);
                runtime_rec(i_exec) = toc;
                if ~any(isnan(R2_rec(:, i_exec)))
                    break
                end
            end
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 13
            %% Nonlinear model based on minimum mean squared error estimator
            model_summary_rec{i_exec} = 'MMSE (scalar)';
            n_AR_lags = 15;
            N_pdf = 300;
            rel_sigma = 7e-3;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                nonlinear_MMSE(Y, n_AR_lags, N_pdf, rel_sigma, k, MMSE_memory, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        
    end
end

%% Choosing the best model (among all but pairwise methods)
if nargout >= 9
    R2_cmp = zeros(n_method);
    for i_method = 1:n_method
        for j_method = setdiff(1:n_method, i_method)
            R2_cmp(i_method, j_method) = ranksum(R2_rec(:, i_method), R2_rec(:, j_method), 'tail', 'right'); % One-sided ranksum test checking if R2_rec(:, i_method) has a significantly larger median than R2_rec(:, j_method)
        end
    end
    best_method = find(all(R2_cmp < 0.5, 2));                               % The best model corresponds to a row with all entries less than 0.5 (at least as good as any other model).
    best_model.model = model_rec{best_method};
    best_model.R2 = R2_rec(:, best_method);
    best_model.whiteness = whiteness_rec(best_method);
    best_model.Y_hat = Y_hat_rec{best_method};
    best_model.runtime = runtime_rec(best_method);
else
    best_model = [];
end