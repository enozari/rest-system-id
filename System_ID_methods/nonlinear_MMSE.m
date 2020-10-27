function [model, R2, whiteness_p, Y_hat] = nonlinear_MMSE(Y, n_AR_lags, N_pdf, rel_sigma, memory, ...
    use_parallel, test_range)
%NONLINEAR_PAIRWISE_MMSE Fitting and cross-validating the optimal minimum
% mean squared error nonlinear estimator.
%
%   Input arguments
% 
%   Y: a data matrix or cell array of data matrices used for system
%   identification. Each element of Y (or Y itself) is one scan, with
%   channels along the first dimension and time along the second dimension.
%   This is the only mandatory input.
% 
%   N_pdf: the number of discretization points for estimating the
%   conditional distributions.
% 
%   pdf_weight: struct variable determining the pdf weighting method and
%   parameter to be used. See MMSE_est.m for details.
% 
%   memory: code determining the amount of memory to be used for MMSE
%   estimation. See MMSE_est.m for details.
% 
%   use_parallel: whether to use parallel loops (parfor) to speed up
%   computations.
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
%   whiteness_p: an n x 1 vector containing the p-values of the chi-squared
%   test of whiteness for the cross-validated residuals of each channel.
% 
%   Y_hat: a cell array the same size as Y but for cross-validated one-step
%   ahead predictions using the fitted model. This is only meaningful for
%   the testing time points, so the entries corresponding to training time
%   points are all NaNs. Also, since each element of Y is a separate scan,
%   its first time point cannot be predicted since no "previous time point"
%   data is available. Therefore, the first column of all elements of Y_hat
%   are also NaNs, regardless of being a training or a test time point.
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

if nargin < 2 || isempty(n_AR_lags)
    n_AR_lags = 10;
end
if nargin < 3 || isempty(N_pdf)
    N_pdf = 300;
end
if nargin < 4 || isempty(rel_sigma)
    rel_sigma = 0.002;
end
if nargin < 5
    memory = [];
end
if nargin < 6 || isempty(use_parallel)
    use_parallel = 1;
end
if nargin < 7 || isempty(test_range)
    test_range = [0.8 1];
end

%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);

AR_lags = 1:n_AR_lags;
Y_train_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, 1+n_AR_lags-lag:end-lag), ...
    permute(AR_lags, [1 3 2]), 'UniformOutput', 0)), Y_train_cell, 'UniformOutput', 0));               % A large matrix of training Y and all its time lags used for regression
Y_train_plus = cell2mat(cellfun(@(Y)Y(:, 1+max(1, n_AR_lags):end), Y_train_cell, 'UniformOutput', 0)); % _plus regers to the time point +1. For instance, each column of Y_train_plus corresponds to one time step after the corresponding column of Y_train.
Y_train = cell2mat(cellfun(@(Y)Y(:, max(1, n_AR_lags):end-1), Y_train_cell, 'UniformOutput', 0));
Y_train_diff = Y_train_plus - Y_train;

n_additional_AR_lags = n_AR_lags - 1;                                       % This is the number of addtional zeros that need to be added to the beginning of Y_test to allow for one step ahead prediction of time points 2 and onwards.
Y_test_cell = cellfun(@(Y)[zeros(n, n_additional_AR_lags), Y], Y_test_cell, 'UniformOutput', 0);
Y_test_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, 1+n_AR_lags-lag:end-lag), ...
    permute(AR_lags, [1 3 2]), 'UniformOutput', 0)), Y_test_cell, 'UniformOutput', 0));                % Similar to Y_train_lags but for test data. Same below.
Y_test = cell2mat(cellfun(@(Y)Y(:, max(1, n_AR_lags):end-1), Y_test_cell, 'UniformOutput', 0));

%% MMSE prediction
Y_test_diff_hat = permute(MMSE_est_nd(permute(Y_train_lags, [2 3 1]), permute(Y_train_diff, [2 3 1]), ...
    permute(Y_test_lags, [2 3 1]), N_pdf, rel_sigma, memory), [3 1 2]);     % Prediction of the derivative (approximated by first difference) of Y_test. The first dimension is predicted-channel, the second dimension is time, and the third dimension is predictor-channel.
Y_test_plus_hat = Y_test + Y_test_diff_hat;                                 % _plus refers to the fact that each column of Y_test_plus_hat corresponds to one time step later than the corresponding column in Y_test. This is corrected by adding a column of NaNs at the beginning to obtain Y_test_hat.

model.eq = ['$y_i(t) - y_i(t-1) = E[y_i(t) - y_i(t-1) | y_i(t-1), y_i(t-2), ..., y_i(t-d)], i = 1,\dots,n$ \\ ' ...
    '``model on demand": no explicit form, $y_i(t) - y_i(t-1)$ estimated separately for any given test (query) point $(y_i(t-1), y_i(t-2), ..., y_i(t-d))$.'];
model.d = n_AR_lags;

%% Output arguments
Y_test_hat = cell2mat(cellfun(@(Y)[nan(n, 1), Y], mat2cell(Y_test_plus_hat, n, N_test_vec-1), 'UniformOutput', 0));
Y_hat = [nan(n, test_ind(1)), Y_test_hat, nan(n, N-test_ind(end))];         % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind));
end

Y_test_plus = cell2mat(cellfun(@(Y)Y(:, 1+max(1, n_AR_lags):end), Y_test_cell, 'UniformOutput', 0));
E_test = Y_test_plus_hat - Y_test_plus;                                     % Prediction error
R2 = 1 - sum(E_test.^2, 2) ./ sum((Y_test_plus - mean(Y_test_plus, 2)).^2, 2);
whiteness_p = my_whitetest(E_test');