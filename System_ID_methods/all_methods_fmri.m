function [model_summary_rec, R2_rec, R2_pw_rec, runtime_rec, whiteness_rec, whiteness_pw_rec, ...
    model_rec, Y_hat_rec, best_model] = all_methods_fmri(Y, Y_pw, TR, k, test_range, MMSE_memory, subj_i_cv)
%ALL_METHODS_FMRI The function that calls all the methods of system
% identification for fMRI data discussed in the paper
% E. Nozari et. al., "Is the brain macroscopically linear? A system
% identification of resting state dynamics", 2021.
%
%   Input Arguments
% 
%   Y: a data matrix or cell array of data matrices. Each element of Y (or
%   Y itself) is one resting state scan, with channels along the first
%   dimension and time along the second dimension. The data in Y is used
%   only for brain-wise methods. This is the only mandatory input.
% 
%   Y_pw: Similar to Y but for pairwise methods. Pairwise methods include
%   the MMSE estimator which is very computationally intensive while at the
%   same time, pairwise methods require less data due to operating in 2
%   dimensions at a time, so it is advised to use less data in Y_pw unless
%   computational burden is not an issue.
% 
%   TR: Sampling time. 
% 
%   k: number of multi-step ahead predictions for cross-validation.
% 
%   test_range: a sub-interval of [0, 1] indicating the portion of Y that
%   is used for test (cross-validation). The rest of Y is used for
%   training.
% 
%   MMSE_memory: The memory code used for MMSE estimation. See MMSE_est.m
%   for details. The advised value is minus the GB of available memory.
% 
%   subj_i_cv: the combination of HCP subject id (subj) and
%   cross-validation fold index (i_cv) for the current data in Y/Y_pw. This
%   is only used in the MLP method to track the time spent running it and
%   recognize hung instances to abort and re-run (both done via main_fmri).
% 
%   Output Arguments
% 
%   model_summary_rec: a cell array of (n_method + n_method_pw) character
%   vectors describing each method in short. n_method is the number of
%   brain-wise methods and n_method_pw is the number of pairwise methods.
% 
%   R2_rec: an n x n_method array where n is the number of brain regions.
%   Each element contains the cross-validated R^2 for that region under
%   that method.
% 
%   R2_pw_rec: an n x n x n_method_pw array similar to R2_rec but for
%   pairwise methods. Due to pairwise prediction, each channel does not
%   have 1 prediction as in brain-wise methods, but instead has n
%   predictions using each other channel. Therefore, R2_pw_rec(i, j, k)
%   contains the R^2 of the prediction of channel i using data from channel
%   j and the pairwise method number k.
% 
%   runtime_rec: an (n_method + n_method_pw) x 1 vector containing the time
%   that each method takes to run.
% 
%   whiteness_rec: an struct object ob size 1 x n_method, containing the
%   statistic (Q) and the randomization-basd significance threshold and
%   p-value of the multivariate whiteness test for each method.
% 
%   whiteness_pw_rec: similar to R2_pw_rec but for whiteness statistics and
%   thresholds for the univariate tests of whiteness for the pairwise
%   methods.
% 
%   model_rec: a cell array of models. Each element of model_rec is a
%   struct with detailed description (functional form and parameters) of
%   the fitted model from any of the model families.
% 
%   Y_hat_rec: an (n_method + n_method_pw) x 1 cell array of predicted time
%   series. Each element of Y_hat_rec is itself a cell array the same size
%   as Y (if that method is brain-wise) or Y_pw (if that method is
%   pairwise).
%   
%   best_model: a struct containing the model, R^2, etc. for the best
%   model. The best model is the one whose R^2 has the largest median, as
%   compared using a ranksum test.
% 
%   Copyright (C) 2021, Erfan Nozari
%   All rights reserved.

if nargin < 2 || isempty(Y_pw)
    Y_pw = Y;
end
if nargin < 3 || isempty(TR)
    warning('Selecting HCP TR = 0.72 by default. Provide its correct value if different.')
    TR = 0.72;
end
if nargin < 4 || isempty(k)
    k = 1;
end
if nargin < 5 || isempty(test_range)
    test_range = [0.8 1];
end
if nargin < 6
    MMSE_memory = [];
end
if nargin < 7 || isempty(subj_i_cv)
    subj_i_cv = '0';
end

%% Initializing the record-keeping variables
n_method = 16;
n_method_pw = 2;
exec_order = 1:(n_method+n_method_pw);                                      % The order in which the methods are run.
model_rec = cell(n_method+n_method_pw, 1);
if iscell(Y)
    n = size(Y{1}, 1);
else
    n = size(Y, 1);
end
R2_rec = zeros(n, n_method);
R2_pw_rec = zeros(n, n, n_method_pw);
whiteness_rec = struct('p', cell(1, n_method), 'stat', [], 'sig_thr', []);
whiteness_pw_rec = struct('p', cell(1, n_method_pw), 'stat', []);
Y_hat_rec = cell(n_method+n_method_pw, 1);
model_summary_rec = cell(n_method+n_method_pw, 1);
runtime_rec = nan(n_method+n_method_pw, 1);

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
    i_method_pw = i_exec - n_method;                                        % This might become negative if the current method is not a pairwise method, which doesn't matter. It will be only used when the current method is a pairwise method, in which case it gives the index of the method.
    switch i_exec
        case 1
            %% The zero model
            model_summary_rec{i_exec} = 'Zero';
            include_W = 0;                                                  % Whether to include network interconnections (effecitive connectivity), not including self-loops
            n_AR_lags = 0;                                                  % Number of autoregressive lags
            W_mask = [];                                                    % The default sparsity structure of linear_BOLD (which is irrelevant here since include_W = 0)
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
            W_mask = 0.95;                                               % LASSO regularization to promote sparsity in effective connectivity
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 4
            %% Linear model at the BOLD level with sparse effective connectivity via LASSO regularization and second order auto-regression
            model_summary_rec{i_exec} = 'VAR-2 (sparse)';
            include_W = 2;
            n_AR_lags = 2;
            W_mask = 0.9;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 5
            %% Linear model at the BOLD level with sparse effective connectivity via LASSO regularization and second order auto-regression
            model_summary_rec{i_exec} = 'AR-2 (sparse)';
            include_W = 1;
            n_AR_lags = 2;
            W_mask = 0.95;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 6
            %% Linear model at the BOLD level with sparse effective connectivity via LASSO regularization and third order auto-regression
            model_summary_rec{i_exec} = 'VAR-3 (sparse)';
            include_W = 2;
            n_AR_lags = 3;
            W_mask = 0.35;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 7
            %% Linear model at the BOLD level with sparse effective connectivity via LASSO regularization and third order auto-regression
            model_summary_rec{i_exec} = 'AR-3 (sparse)';
            include_W = 1;
            n_AR_lags = 3;
            W_mask = 0.5;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 8
            %% Linear model at the neural level
            model_summary_rec{i_exec} = 'Linear w/ HRF';
            n_h = 5;
            n_phi = 5;
            n_psi = 5;
            W_mask = 11;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_neural(Y, TR, n_h, n_phi, n_psi, W_mask, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 9
            %% Linear model via subspace identification
            model_summary_rec{i_exec} = 'Subspace';
            s = 1;
            r = 3;
            n = 25;
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                linear_subspace(Y, s, r, n, k, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 10
            %% Nonlinear model via MINDy [Singh et al., 2019]
            model_summary_rec{i_exec} = 'NMM';
            doPreProc = 'n';
            lambda = {};
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                nonlinear_MINDy(Y, TR, doPreProc, lambda, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 11
            %% Nonlinear model via MINDy [Singh et al., 2019] applied directly to BOLD data (no HRF/deconvolution)
            model_summary_rec{i_exec} = 'NMM w/ HRF';
            doPreProc = 'y';
            lambda = {};
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                nonlinear_MINDy(Y, TR, doPreProc, lambda, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 12
            %% Nonlinear model based on locally linear manifold learning
            model_summary_rec{i_exec} = 'Manifold';
            n_AR_lags = 1;
            kernel = 'Gaussian';
            h = 830;
            scale_h.do_scale = true;
            scale_h.base_med_dist = 565;                                    % The median train-test distance in the original dataset over which h was optimized.
            tic
            [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                nonlinear_manifold(Y, n_AR_lags, kernel, h, scale_h, k, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 13
            %% Nonlinear model based on deep neural networks
            model_summary_rec{i_exec} = 'MLP';
            n_AR_lags = 1;
            hidden_width = 2;
            hidden_depth = 6;
            dropout_prob = 0.5;
            lambda = 1e-4;
            if use_parallel
                exe_env = 'auto';                                           % The 'ExecutionEnvironment' option of the neural network toolbox
            else
                exe_env = 'cpu';
            end
            
            dnn_dt = datetime;
            save(['main_data_dnn/' subj_i_cv '_dnn.mat'], 'dnn_dt')
            
            for i = 3:10
                learn_rate = 10^-i
                tic
                [model_rec{i_exec}, R2_rec(:, i_exec), whiteness_rec(i_exec), Y_hat_rec{i_exec}] = ...
                    nonlinear_MLP(Y, n_AR_lags, hidden_width, hidden_depth, dropout_prob, ...
                    exe_env, learn_rate, lambda, k, test_range);
                runtime_rec(i_exec) = toc;
                if ~any(isnan(R2_rec(:, i_exec)))
                    break
                end
            end
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 14
            %% Nonlinear model based on convolutional neural networks
            model_summary_rec{i_exec} = 'CNN';
            n_AR_lags = 17;
            hidden_depth = 2;
            filter_size = 7;
            n_filter = 11;
            pool_size = 4;
            dilation_factor = [];
            exp_dilation = [];
            dropout_prob = 0.4;
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
        case 15
            %% Nonlinear model based on LSTM neural networks with infinite impulse response
            model_summary_rec{i_exec} = 'LSTM1';
            rec_layer = 'lstm';
            n_AR_lags = nan;
            hidden_width = 12;
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
        case 16
            %% Nonlinear model based on LSTM neural networks with finite impulse response
            model_summary_rec{i_exec} = 'LSTM2';
            rec_layer = 'lstm';
            n_AR_lags = 1;
            hidden_width = 16;
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
        
        case 17
            %% Linear model at the BOLD level via pairwise regression
            model_summary_rec{i_exec} = 'Linear (pairwise)';
            tic
            [model_rec{i_exec}, R2_pw_rec(:, :, i_method_pw), whiteness_pw_rec(i_method_pw), ...
                Y_hat_rec{i_exec}] = linear_pairwise(Y_pw, k, use_parallel, test_range);
            runtime_rec(i_exec) = toc;
            disp([model_summary_rec{i_exec} ' completed in ' num2str(runtime_rec(i_exec)) ' seconds.'])
        case 18
            %% Nonlinear model via MMSE
            model_summary_rec{i_exec} = 'MMSE (pairwise)';
            N_pdf = 280;
            pdf_weight.method = 'normpdf';
            pdf_weight.rel_sigma = 0.156;
            tic
            [model_rec{i_exec}, R2_pw_rec(:, :, i_method_pw), whiteness_pw_rec(i_method_pw), ...
                Y_hat_rec{i_exec}] = nonlinear_pairwise_MMSE(Y_pw, N_pdf, pdf_weight, MMSE_memory, ...
                k, use_parallel, test_range);
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