function [model_summary_rec, R2_rec, runtime_rec, whiteness_p_rec, model_rec, Y_hat_rec, ...
    best_model] = all_methods_ieeg(Y, TR, test_range, MMSE_memory)
%ALL_METHODS_IEEG The function that calls all the methods of system
% identification for iEEG data discussed in the paper
% E. Nozari et. al., "Is the brain macroscopically linear? A system
% identification of resting state dynamics", 2020.
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
%   test_range: a sub-interval of [0, 1] indicating the portion of Y that
%   is used for test (cross-validation). The rest of Y is used for
%   training.
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
%   whiteness_p_rec: an array the same size as R2_rec and with similar
%   structure, except that each element contains the p-value of the
%   chi-squared test of whiteness for the residuals of cross-valudated
%   prediction of that channel under that method.
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
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

if nargin < 2 || isempty(TR)
    warning('Selecting TR = 0.002 by default. Provide its correct value if different.')
    TR = 0.002;
end
if nargin < 3 || isempty(test_range)
    test_range = [0.8 1];
end
if nargin < 4
    MMSE_memory = [];
end

%% Initializing the record-keeping variables
n_method = 10;
exec_order = 1:n_method;                                                    % The order in which the methods are run. If run on a cluster with license limitations, using exec_order = [1 2 5 6 8 10 3 4 7 9] can be more efficient.
model_rec = cell(n_method, 1);
if iscell(Y)
    n = size(Y{1}, 1);
else
    n = size(Y, 1);
end
R2_rec = zeros(n, n_method);
whiteness_p_rec = zeros(n, n_method);
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
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_p_rec(:, i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 2
            %% Simple linear model with full effective connectivity
            model_summary_rec{i_exec} = 'Linear (dense)';
            include_W = 1;
            n_AR_lags = 1;
            W_mask = 'full';                                                % Dense, potentially all-to-all effective connectivity
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_p_rec(:, i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 3
            %% Simple linear model with sparse effective connectivity via LASSO regularization
            model_summary_rec{i_exec} = 'Linear (sparse)';
            include_W = 1;
            n_AR_lags = 1;
            W_mask = 1.2;                                                    % LASSO regularization to promote sparsity in effective connectivity, with lambda parameter equal to 1.2
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_p_rec(:, i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 4
            %% AR linear model with sparse effective connectivity via LASSO regularization and 100 AR lags
            model_summary_rec{i_exec} = 'AR-100 (sparse)';
            include_W = 1;
            n_AR_lags = 100;
            W_mask = 1.5;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_p_rec(:, i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 5
            %% Scalar AR linear model with sparse effective connectivity via LASSO regularization and ~100 AR lags
            model_summary_rec{i_exec} = 'AR-100 (scalar)';
            include_W = 0;
            n_AR_lags = 102;
            W_mask = [];
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_p_rec(:, i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 6
            %% Linear model via subspace identification
            model_summary_rec{i_exec} = 'Subspace';
            s = 11;
            r = 49;
            n = 436;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_p_rec(:, i_exec), Y_hat_rec{i_exec}] = ...
                linear_subspace(Y, s, r, n, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 7
            %% Nonlinear model via MINDy [Singh et al., 2019] applied directly to iEEG time series (no HRF/deconvolution)
            model_summary_rec{i_exec} = 'NMM';
            lambda = {0.2 0.2 2 0.5};
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_p_rec(:, i_exec), Y_hat_rec{i_exec}] = ...
                nonlinear_MINDy(Y, TR, 'n', lambda, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 8
            %% Nonlinear model based on locally linear manifold learning
            model_summary_rec{i_exec} = 'Manifold';
            n_AR_lags = 7;
            kernel = 'Gaussian';
            h = 1.2e4;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_p_rec(:, i_exec), Y_hat_rec{i_exec}] = ...
                nonlinear_manifold(Y, n_AR_lags, kernel, h, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 9
            %% Nonlinear model based on deep neural networks
            model_summary_rec{i_exec} = 'DNN';
            n_AR_lags = 6;
            hidden_width = 26;
            hidden_depth = 4;
            if use_parallel
                exe_env = 'auto';                                           % The 'ExecutionEnvironment' option of the neural network toolbox
            else
                exe_env = 'cpu';
            end
            for i = 1:5
                InitialLearnRate_drop_factor = 10^-i;
                tic
                [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_p_rec(:, i_exec), Y_hat_rec{i_exec}] = ...
                    nonlinear_DNN(Y, n_AR_lags, hidden_width, hidden_depth, exe_env, ...
                    InitialLearnRate_drop_factor, test_range);
                runtime_rec(i_exec) = toc;
                if ~any(isnan(R2_rec(:, i_exec)))
                    break
                end
            end
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 10
            %% Nonlinear model via MMSE on a scalar basis
            model_summary_rec{i_exec} = 'MMSE (scalar)';
            n_AR_lags = 15;
            N_pdf = 300;
            rel_sigma = 7e-3;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_p_rec(:, i_exec), Y_hat_rec{i_exec}] = ...
                nonlinear_MMSE(Y, n_AR_lags, N_pdf, rel_sigma, MMSE_memory, use_parallel, test_range);
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
    best_model.whiteness_p = whiteness_p_rec(:, best_method);
    best_model.Y_hat = Y_hat_rec{best_method};
    best_model.runtime = runtime_rec(best_method);
else
    best_model = [];
end