function [model, R2, whiteness, Y_hat] = nonlinear_CNN(Y, n_AR_lags, hidden_depth, ...
    filter_size, n_filter, pool_size, dilation_factor, exp_dilation, dropout_prob, ...
    exe_env, learn_rate, test_range)
%NONLINEAR_CNN Fitting and cross-validating a nonlinear model using ReLU
% CNN deep neural networks.
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
%   hidden_depth: the depth of the network (number of hidden layers)
% 
%   filter_size: the size (number of impulse response samples) for the
%   convoluation filters in all layers.
% 
%   n_filter: the number of filters (a.k.a. number of channels) in each
%   layer.
% 
%   pool_size: the size of the average pooling layers throughout the
%   network.
% 
%   dilation_factor: the dilation factor for all of the convolutional
%   layers in the network. Note that this will have a slight different
%   meaning based on the value of exp_dilation.
% 
%   exp_dilation: binary flag indicating whether exponentially increasing
%   dilation (conventional for CNNs with temporal convolution) should be
%   applied. If exp_dilation == 0, then dilation_factor is directly the
%   dilation factor of the convolutional layers. If exp_dilation == 1, then
%   dilation_factor^(i-1) is the dilation factor at layer i.
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
    n_AR_lags = 3;
end
if nargin < 3 || isempty(hidden_depth)
    hidden_depth = 3;
end
if nargin < 4 || isempty(filter_size)
    filter_size = 3;
end
if nargin < 5 || isempty(n_filter)
    n_filter = 10;
end
if nargin < 6 || isempty(pool_size)
    pool_size = 1;
end
if nargin < 7 || isempty(dilation_factor)
    dilation_factor = 1;
end
if nargin < 8 || isempty(exp_dilation)
    exp_dilation = false;
end
if nargin < 9 || isempty(dropout_prob)
    dropout_prob = 0.5;
end
if nargin < 10 || isempty(exe_env)
    exe_env = 'auto';
end
if nargin < 11 || isempty(learn_rate)
    learn_rate = 1e-3;
end
if nargin < 12 || isempty(test_range)
    test_range = [0.8 1];
end

%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);
mu = mean(cell2mat(Y_train_cell), 2);
sigma = std(cell2mat(Y_train_cell), [], 2);
Y_train_norm_cell = cellfun(@(Y)(Y - mu) ./ sigma, Y_train_cell, 'UniformOutput', 0);
Y_test_norm_cell = cellfun(@(Y)(Y - mu) ./ sigma, Y_test_cell, 'UniformOutput', 0);

AR_lags = 1:n_AR_lags;
Y_train_norm_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, 1+n_AR_lags-lag:end-lag), ...
    permute(AR_lags, [1 3 2]), 'UniformOutput', 0)), Y_train_norm_cell, 'UniformOutput', 0));               % A large matrix of training Y and all its time lags used for regression
Y_train_norm_plus = cell2mat(cellfun(@(Y)Y(:, 1+max(1, n_AR_lags):end), Y_train_norm_cell, 'UniformOutput', 0)); % _plus regers to the time point +1. For instance, each column of Y_train_plus corresponds to one time step after the corresponding column of Y_train.
Y_train_norm = cell2mat(cellfun(@(Y)Y(:, max(1, n_AR_lags):end-1), Y_train_norm_cell, 'UniformOutput', 0));
Y_train_norm_diff = Y_train_norm_plus - Y_train_norm;
N_train = size(Y_train_norm_lags, 2);

%% Constructing and training the network
layers = imageInputLayer([1 n_AR_lags n]);
try
    for i = 1:hidden_depth
        layers = [layers, convolution2dLayer([1 filter_size], n_filter, 'Padding', 'same', ...
            'DilationFactor', dilation_factor*~exp_dilation+dilation_factor^(i-1)*exp_dilation), ...
            batchNormalizationLayer, reluLayer, averagePooling2dLayer([1 pool_size])];
    end
catch
    for i = 1:hidden_depth
        layers = [layers, convolution2dLayer([1 filter_size], n_filter, 'Padding', 'same'), ...
            batchNormalizationLayer, reluLayer, averagePooling2dLayer([1 pool_size])];
    end
end
layers = [layers, dropoutLayer(dropout_prob), fullyConnectedLayer(n), regressionLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', learn_rate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'none', ...
    'Verbose', false, ...
    'ExecutionEnvironment', exe_env);
net = trainNetwork(permute(Y_train_norm_lags, [4 3 1 2]), Y_train_norm_diff', layers, options);
    
model.eq = ['$y(t) - y(t-1) = f(y(t-1), ..., y(t-d)) + e(t)$ \\ ' ...
    'f(\cdot) is a deep convolutional neural network.'];
model.f = net;
model.d = n_AR_lags;

%% Cross-validated one step ahead prediction
n_additional_AR_lags = n_AR_lags - 1;                                       % This is the number of addtional zeros that need to be added to the beginning of Y_test to allow for one step ahead prediction of time points 2 and onwards.
Y_test_norm_cell = cellfun(@(Y)[zeros(n, n_additional_AR_lags), Y], Y_test_norm_cell, 'UniformOutput', 0);
Y_test_norm_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, 1+n_AR_lags-lag:end-lag), ...
    permute(AR_lags, [1 3 2]), 'UniformOutput', 0)), Y_test_norm_cell, 'UniformOutput', 0));                % Similar to Y_train_lags but for test data. Same below.
Y_test_norm = cell2mat(cellfun(@(Y)Y(:, max(1, n_AR_lags):end-1), Y_test_norm_cell, 'UniformOutput', 0));

Y_test_norm_diff_hat = predict(net, permute(Y_test_norm_lags, [4 3 1 2]), 'ExecutionEnvironment', exe_env)'; % DNN's prediction of the derivative (approximated by first difference) of Y_test
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