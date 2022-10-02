function [model, R2, whiteness, Y_hat, runtime] = nonlinear_LSTM(Y, rec_layer, n_AR_lags, ...
    hidden_width, exe_env, learn_rate, k, test_range)
%NONLINEAR_LSTM Fitting and cross-validating a nonlinear model using
% LSTM/GRU neural networks.
%
%   Input arguments
% 
%   Y: a data matrix or cell array of data matrices used for system
%   identification. Each element of Y (or Y itself) is one scan, with
%   channels along the first dimension and time along the second dimension.
%   This is the only mandatory input.
%     
%   rec_layer: either 'lstm' for LSTM recurrent layer or 'gru' for GRU
%   recurrent layer.
% 
%   n_AR_lags: number of past (auto-regressive) lags of Y that are used to
%   predict the next time step. LSTM can be used for sequence-to-sequence
%   and sequence-to-one predictions. For the former, set n_AR_lags = nan as
%   the entire sequence is predicted one step at a time.
% 
%   hidden_width: the width of the network (number of LSTM units)
% 
%   exe_env: the value of the 'ExecutionEnvironment' option of the
%   trainNetwork function. Options are 'auto', 'cpu', 'gpu', 'multi-gpu',
%   or 'parallel'. See documentation for trainingOptions for details.
% 
%   learn_rate: the initial learning rate.
% 
%   k: number of multi-step ahead predictions for cross-validation.
% 
%   test_range: a sub-interval of [0, 1] indicating the portion of Y that
%   is used for test (cross-validation). The rest of Y is used for
%   training.
% 
%   Output arguments
% 
%   model: a struct with detailed description (functional form and
%   parameters) of the fitted model.
% 
%   R2: an n x 1 vector containing the cross-validated prediction R^2 of
%   the n channels.
% 
%   whiteness: a struct containing the statistic (Q) and the
%   randomization-basd significance threshold and p-value of the
%   multivariate whiteness test.
% 
%   Y_hat: a cell array the same size as Y but for cross-validated one-step
%   ahead predictions using the fitted model. This is only meaningful for
%   the testing time points, so the entries corresponding to training time
%   points are all NaNs. Also, since each element of Y is a separate scan,
%   its first time point cannot be predicted since no "previous time point"
%   data is available. Therefore, the first column of all elements of Y_hat
%   are also NaNs, regardless of being a training or a test time point.
% 
%   Copyright (C) 2021, Erfan Nozari
%   All rights reserved.

if nargin < 2 || isempty(rec_layer)
    rec_layer = 'lstm';
end
if nargin < 3 || isempty(n_AR_lags)
    n_AR_lags = nan;
end
if nargin < 4 || isempty(hidden_width)
    hidden_width = 10;
end
if nargin < 5 || isempty(exe_env)
    exe_env = 'auto';
end
if nargin < 6 || isempty(learn_rate)
    learn_rate = 0.005;
end
if nargin < 7 || isempty(k)
    k = 1;
end
if nargin < 8 || isempty(test_range)
    test_range = [0.8 1];
end

%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);

runtime_train_start = tic;

n_test = numel(Y_test_cell);
mu = mean(cell2mat(Y_train_cell), 2);
sigma = std(cell2mat(Y_train_cell), [], 2);
Y_train_norm_cell = cellfun(@(Y)(Y - mu) ./ sigma, Y_train_cell, 'UniformOutput', 0);
Y_test_norm_cell = cellfun(@(Y)(Y - mu) ./ sigma, Y_test_cell, 'UniformOutput', 0);

%% Constructing and training the network
if isnan(n_AR_lags)
    Y_train_norm_0_cell = cellfun(@(Y)Y(:, 1:end-1), Y_train_norm_cell, 'UniformOutput', 0);
    Y_train_norm_diff_cell = cellfun(@(Y)diff(Y, 1, 2), Y_train_norm_cell, 'UniformOutput', 0);
    switch rec_layer
        case 'lstm'
            layers = [sequenceInputLayer(n), lstmLayer(hidden_width), fullyConnectedLayer(n), regressionLayer];
        case 'gru'
            layers = [sequenceInputLayer(n), gruLayer(hidden_width), fullyConnectedLayer(n), regressionLayer];
    end
    options = trainingOptions('adam', ...
        'MaxEpochs', 250, ...
        'GradientThreshold', 1, ...
        'InitialLearnRate', learn_rate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 125, ...
        'LearnRateDropFactor', 0.2, ...
        'Verbose', 0, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', exe_env);
    net = trainNetwork(Y_train_norm_0_cell', Y_train_norm_diff_cell', layers, options);
    model.eq = ['$y(t) = f(y(t-1)) + e(t)$ \\ ' ...
        'f(\cdot) is an LSTM neural network.'];
    model.f = net;
else
    Y_train_norm_lags_cell = cellfun(@(Y)arrayfun(@(t)Y(:, t-n_AR_lags:t-1), ...
        n_AR_lags+1:size(Y, 2), 'UniformOutput', 0), Y_train_norm_cell, 'UniformOutput', 0);
    Y_train_norm_lags_cell = [Y_train_norm_lags_cell{:}]';
    Y_train_norm_diff_cell = cell2mat(cellfun(@(Y)diff(Y(:, n_AR_lags:end), 1, 2), ...
        Y_train_norm_cell, 'UniformOutput', 0))';
    try
        eval(['layers = [sequenceInputLayer(n), ' ...
            rec_layer 'Layer(hidden_width, ''OutputMode'', ''last'', ' ...
            '''HiddenState'', zeros(hidden_width, 1)), fullyConnectedLayer(n), regressionLayer];'])
    catch ME
        if isequal(ME.identifier, 'MATLAB:InputParser:UnmatchedParameter') % Older versions of MATLAB do not allow for setting 'HiddenState'
            eval(['layers = [sequenceInputLayer(n), ' ...
                rec_layer 'Layer(hidden_width, ''OutputMode'', ''last''), ' ...
                'fullyConnectedLayer(n), regressionLayer];'])
        else
            rethrow(ME)
        end
    end
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'GradientThreshold', 1, ...
        'InitialLearnRate', learn_rate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 125, ...
        'LearnRateDropFactor', 0.2, ...
        'Verbose', 0, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', exe_env);
    net = trainNetwork(Y_train_norm_lags_cell, Y_train_norm_diff_cell, layers, options);
    model.eq = ['$y(t) = f(y(t-1, ..., t-d)) + e(t)$ \\' ...
        'f(\cdot) is an LSTM neural network.'];
    model.f = net;
end

runtime.train = toc(runtime_train_start);

%% Cross-validated k-step ahead prediction
runtime_test_start = tic;

Y_test_norm_hat_cell = cellfun(@(Y)nan(size(Y)), Y_test_norm_cell, 'UniformOutput', 0);
if isnan(n_AR_lags)    
    for i_test = 1:n_test
        for t = k+1:N_test_vec(i_test)
            net = rebootState(net, Y_train_norm_0_cell, exe_env);
            for tau = 1:t-k
                [net, Y_test_norm_diff_hat] = predictAndUpdateState(net, ...
                    Y_test_norm_cell{i_test}(:, tau), 'ExecutionEnvironment', exe_env, ...
                    'MiniBatchSize', 1);
            end
            Y_test_norm_hat_tau = Y_test_norm_cell{i_test}(:, tau) + Y_test_norm_diff_hat;
            for tau = t-k+1:t-1
                [net, Y_test_norm_diff_hat] = predictAndUpdateState(net, ...
                    Y_test_norm_hat_tau, 'ExecutionEnvironment', exe_env, ...
                    'MiniBatchSize', 1);
                Y_test_norm_hat_tau = Y_test_norm_hat_tau + Y_test_norm_diff_hat;
            end
                Y_test_norm_hat_cell{i_test}(:, t) = Y_test_norm_hat_tau;
        end
    end
else
    n_add_AR_lags = n_AR_lags - 1;
    Y_test_norm_cell = cellfun(@(Y)[zeros(n, n_add_AR_lags) Y], Y_test_norm_cell, 'UniformOutput', 0);
    for i_test = 1:n_test
        for t = k+1:N_test_vec(i_test)
            net = resetState(net);
            Phi = Y_test_norm_cell{i_test}(:, n_add_AR_lags+(t-k-n_AR_lags+1:t-k));
            for i = 1:k
                [net, Y_test_norm_diff_hat] = predictAndUpdateState(net, ...
                    Phi, 'ExecutionEnvironment', exe_env, 'MiniBatchSize', 1);
                Y_test_norm_hat = Phi(:, end) + Y_test_norm_diff_hat';
                Phi = [Phi(:, 2:end), Y_test_norm_hat];
            end
            Y_test_norm_hat_cell{i_test}(:, t) = Y_test_norm_hat;
        end
    end
end

Y_test_hat_cell = cellfun(@(Y)sigma .* Y + mu, Y_test_norm_hat_cell, 'UniformOutput', 0);
Y_test_plus_hat = cell2mat(cellfun(@(Y)Y(:, k+1:end), Y_test_hat_cell, 'UniformOutput', 0));
Y_test_plus = cell2mat(cellfun(@(Y)Y(:, k+1:end), Y_test_cell, 'UniformOutput', 0));

Y_hat = [nan(n, test_ind(1)), cell2mat(Y_test_hat_cell), nan(n, N-test_ind(end))];         % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind));
end

runtime.test = toc(runtime_test_start);
runtime.total = runtime.train + runtime.test;

E_test = Y_test_plus_hat - Y_test_plus;                                     % Prediction error
R2 = 1 - sum(E_test.^2, 2) ./ sum((Y_test_plus - mean(Y_test_plus, 2)).^2, 2);
[whiteness.p, whiteness.stat, whiteness.sig_thr] = my_whitetest_multivar(E_test);
end

%% Auxiliary functions
function net = rebootState(net, trainingSeq, exe_env)
net = resetState(net);
net = predictAndUpdateState(net, trainingSeq, 'ExecutionEnvironment', exe_env, ...
    'MiniBatchSize', 1);
end