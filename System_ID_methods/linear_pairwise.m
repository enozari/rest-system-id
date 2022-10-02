function [model, R2, whiteness, Y_hat, runtime] = linear_pairwise(Y, k, use_parallel, test_range)
%LINEAR_PAIRWISE Fitting and cross-validating simple pairwise linear
% regression models directly at the signal level.
%
%   Input arguments
% 
%   Y: a data matrix or cell array of data matrices used for system
%   identification. Each element of Y (or Y itself) is one scan, with
%   channels along the first dimension and time along the second dimension.
%   This is the only mandatory input.
% 
%   k: number of multi-step ahead predictions for cross-validation
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
%   whiteness: a struct containing the matrix of statistics and chi-squared
%   p-values of the univariate whiteness tests for each pair of channels.
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

if nargin < 2 || isempty(k)
    k = 1;
end
if nargin < 3 || isempty(use_parallel)
    use_parallel = 1;
end
if nargin < 4 || isempty(test_range)
    test_range = [0.8 1];
end

%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);

runtime_train_start = tic;

Y_train = cell2mat(cellfun(@(Y)Y(:, 1:end-1), Y_train_cell, 'UniformOutput', 0));
Y_train_diff = cell2mat(cellfun(@(Y)diff(Y, 1, 2), Y_train_cell, 'UniformOutput', 0));
Y_test = cell2mat(cellfun(@(Y)Y(:, 1:end-1), Y_test_cell, 'UniformOutput', 0));

%% Least squares
W = permute(sum(permute(Y_train, [3 2 1]) .* Y_train_diff, 2), [1 3 2]) ./ sum(Y_train.^2, 2)'; % Least squares in closed form

model.eq = '$y_i(t) - y_i(t-1) = W_{ij} y_j(t-1), i,j = 1,\dots,n$';
model.W = W;

runtime.train = toc(runtime_train_start);

%% Cross-validated k-step ahead prediction
runtime_test_start = tic;

Phi = Y_test;
for i = 1:k-1
    Y_test_plus_hat = Phi + diag(W) .* Phi; 
    Phi = Y_test_plus_hat(:, 1:end-1);
end
Y_test_plus_hat = Phi + permute(W, [1 3 2]) .* permute(Phi, [3 2 1]);

Y_test_hat = cell2mat(cellfun(@(Y)[nan(n, k, n), Y], mat2cell(Y_test_plus_hat, n, N_test_vec-k, n), 'UniformOutput', 0));
Y_hat = [nan(n, test_ind(1), n), Y_test_hat, nan(n, N-test_ind(end), n)];   % Adding NaNs for any time point that is used for training, so that Y_hat has the same number of time points as Y.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind), n);
end

runtime.test = toc(runtime_test_start);
runtime.total = runtime.train + runtime.test;

Y_test_plus = cell2mat(cellfun(@(Y)Y(:, k+1:end), Y_test_cell, 'UniformOutput', 0));
E_test = Y_test_plus_hat - Y_test_plus;
R2 = permute(1 - sum((E_test).^2, 2) ./ sum((Y_test_plus - mean(Y_test_plus, 2)).^2, 2), [1 3 2]); % Since this is a pairwise method, R2 is a matrix, whose (i, j) entry is the R2 of predicting y_i(t) from y_j(t-1)

whiteness_p = nan(n);                                                       % This is also a matrix similar to R2
whiteness_stat = nan(n);
if use_parallel
    parfor j = 1:n
        [whiteness_p(:, j), whiteness_stat(:, j)] = my_whitetest(E_test(:, :, j));
    end
else
    for j = 1:n
        [whiteness_p(:, j), whiteness_stat(:, j)] = my_whitetest(E_test(:, :, j));
    end
end
whiteness.p = whiteness_p;
whiteness.stat = whiteness_stat;