function [model, R2, whiteness, Y_hat, runtime] = zero(Y, k, test_range)
%ZERO The Zero model (a.k.a. random walk, naive, zero-order-hold).
%
%   Input arguments
% 
%   Y: a data matrix or cell array of data matrices used for system
%   identification. Each element of Y (or Y itself) is one scan, with
%   channels along the first dimension and time along the second dimension.
%   This is the only mandatory input.
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
%   Copyright (C) 2022, Erfan Nozari
%   All rights reserved.

if nargin < 2 || isempty(k)
    k = 1;
end
if nargin < 3 || isempty(test_range)
    test_range = [0.8 1];
end

%% Organizing data into separate train and test segments
[~, Y_test_cell, break_ind, test_ind, ~, n, N] = tt_decomp(Y, test_range);

model.eq = '$y(t) - y(t-1) = e(t)$';

runtime.train = 0;

%% Cross-validated k-step ahead prediction
runtime_test_start = tic;

Y_test_plus = cell2mat(cellfun(@(Y)Y(:, k+1:end), Y_test_cell, 'UniformOutput', 0));
Y_test_plus_hat = cellfun(@(Y)Y(:, 1:end-k), Y_test_cell, 'UniformOutput', 0);

Y_test_hat = cell2mat(cellfun(@(Y)[nan(n, k), Y], Y_test_plus_hat, 'UniformOutput', 0));
Y_hat = [nan(n, test_ind(1)), Y_test_hat, nan(n, N-test_ind(end))];         % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind));
end

runtime.test = toc(runtime_test_start);
runtime.total = runtime.train + runtime.test;

E_test = cell2mat(Y_test_plus_hat) - Y_test_plus;                                     % Prediction error
R2 = 1 - sum(E_test.^2, 2) ./ sum((Y_test_plus - mean(Y_test_plus, 2)).^2, 2);
[whiteness.p, whiteness.stat, whiteness.sig_thr] = my_whitetest_multivar(E_test);