function [model, R2, whiteness, Y_hat] = nonlinear_MLP(Y, n_AR_lags, hidden_width, hidden_depth, dropout_prob, ...
    exe_env, learn_rate, test_range)
%NONLINEAR_MLP Fitting and cross-validating a nonlinear model using ReLU
% MLP deep neural networks.
%
%   Input arguments
% 
%   Y: a data matrix or cell array of data matrices used for system
%   identification. Each element of Y (or Y itself) is one scan, with
%   channels along the first dimension and time along the second dimension.
%   This is the only mandatory input.
% 
%   n_AR_lags: number of past (auto-regressive) lags of Y to include as
%   input to the DNN
% 
%   hidden_width: the width of the network (number of hidden units per
%   hidden layer)
% 
%   hidden_depth: the depth of the network (number of hidden layers)
% 
%   dropout_prob: probability of dropout at each layer of the network. Set
%   to 0 to deactivate dropout.
% 
%   exe_env: the value of the 'ExecutionEnvironment' option of the
%   trainNetwork function. Options are 'auto', 'cpu', 'gpu', 'multi-gpu',
%   or 'parallel'. See documentation for trainingOptions for details.
% 
%   learn_rate: the initial learning rate.
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

if nargin < 2 || isempty(n_AR_lags)
    n_AR_lags = 1;
end
if nargin < 3 || isempty(hidden_width)
    hidden_width = 2;
end
if nargin < 4 || isempty(hidden_depth)
    hidden_depth = 6;
end
if nargin < 5 || isempty(dropout_prob)
    dropout_prob = 0.5;
end
if nargin < 6 || isempty(exe_env)
    exe_env = 'auto';
end
if nargin < 7 || isempty(learn_rate)
    learn_rate = 1e-3;
end
if nargin < 8 || isempty(test_range)
    test_range = [0.8 1];
end

%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);
mu = mean(cell2mat(Y_train_cell), 2);
sigma = std(cell2mat(Y_train_cell), [], 2);
Y_train_norm_cell = cellfun(@(Y)(Y - mu) ./ sigma, Y_train_cell, 'UniformOutput', 0);
Y_test_norm_cell = cellfun(@(Y)(Y - mu) ./ sigma, Y_test_cell, 'UniformOutput', 0);

AR_lags = 1:n_AR_lags;
Y_train_norm_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, 1+n_AR_lags-lag:end-lag), AR_lags', ...
    'UniformOutput', 0)), Y_train_norm_cell, 'UniformOutput', 0));               % A large matrix of training Y and all its time lags used for regression
Y_train_norm_plus = cell2mat(cellfun(@(Y)Y(:, 1+max(1, n_AR_lags):end), Y_train_norm_cell, 'UniformOutput', 0)); % _plus regers to the time point +1. For instance, each column of Y_train_plus corresponds to one time step after the corresponding column of Y_train.
Y_train_norm = cell2mat(cellfun(@(Y)Y(:, max(1, n_AR_lags):end-1), Y_train_norm_cell, 'UniformOutput', 0));
Y_train_norm_diff = Y_train_norm_plus - Y_train_norm;
N_train = size(Y_train_norm_lags, 2);

%% Constructing and training the network
layers = imageInputLayer([1 1 n*n_AR_lags]);
for i = 1:hidden_depth
    layers = [layers, fullyConnectedLayer(hidden_width), batchNormalizationLayer, reluLayer, ...
        dropoutLayer(dropout_prob)];
end
layers = [layers, fullyConnectedLayer(n), regressionLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', learn_rate, ...
    'MiniBatchSize', round(N_train/10), ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.9, ...
    'LearnRateDropPeriod', 20, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'none', ...
    'ExecutionEnvironment', exe_env, ...
    'Verbose', false);

net = trainNetwork(permute(Y_train_norm_lags, [3 4 1 2]), Y_train_norm_diff', layers, options);
    
model.eq = ['$y(t) - y(t-1) = f(y(t-1)) + e(t)$ \\ ' ...
    'f(\cdot) is a deep MLP neural network.'];
model.f = net;

%% Cross-validated one step ahead prediction
n_additional_AR_lags = n_AR_lags - 1;                                       % This is the number of addtional zeros that need to be added to the beginning of Y_test to allow for one step ahead prediction of time points 2 and onwards.
Y_test_norm_cell = cellfun(@(Y)[zeros(n, n_additional_AR_lags), Y], Y_test_norm_cell, 'UniformOutput', 0);
Y_test_norm_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, 1+n_AR_lags-lag:end-lag), AR_lags', ...
    'UniformOutput', 0)), Y_test_norm_cell, 'UniformOutput', 0));                % Similar to Y_train_lags but for test data. Same below.
Y_test_norm = cell2mat(cellfun(@(Y)Y(:, max(1, n_AR_lags):end-1), Y_test_norm_cell, 'UniformOutput', 0));

Y_test_norm_diff_hat = predict(net, permute(Y_test_norm_lags, [3 4 1 2]), 'ExecutionEnvironment', exe_env)'; % DNN's prediction of the derivative (approximated by first difference) of Y_test
Y_test_norm_plus_hat = Y_test_norm + Y_test_norm_diff_hat;                                 % _plus refers to the fact that each column of Y_test_plus_hat corresponds to one time step later than the corresponding column in Y_test. This is corrected by adding a column of NaNs at the beginning to obtain Y_test_hat.
Y_test_plus_hat = sigma .* Y_test_norm_plus_hat + mu;
Y_test_hat = cell2mat(cellfun(@(Y)[nan(n, 1), Y], mat2cell(Y_test_plus_hat, n, N_test_vec-1), 'UniformOutput', 0));

Y_hat = [nan(n, test_ind(1)), Y_test_hat, nan(n, N-test_ind(end))];         % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind));
end

Y_test_plus = cell2mat(cellfun(@(Y)Y(:, 2:end), Y_test_cell, 'UniformOutput', 0));
E_test = Y_test_plus_hat - Y_test_plus;                                     % Prediction error
R2 = 1 - sum(E_test.^2, 2) ./ sum((Y_test_plus - mean(Y_test_plus, 2)).^2, 2);
[whiteness.p, whiteness.stat, whiteness.sig_thr] = my_whitetest_multivar(E_test);