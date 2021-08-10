function [model, R2, whiteness, Y_hat] = nonlinear_pairwise_MMSE(Y, N_pdf, pdf_weight, memory, ...
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

if nargin < 2 || isempty(N_pdf)
    N_pdf = 200;
end
if nargin < 3 || isempty(pdf_weight)
    pdf_weight.method = 'normpdf';                                          % Can be 'knn' or 'normpdf'
    pdf_weight.rel_sigma = 0.02;                                            % Set pdf_weight.rel_K for pdf_weight.method == 'knn' or pdf_weight.rel_sigma for pdf_weight.method == 'normpdf'.
end
if nargin < 4
    memory = [];
end
if nargin < 5 || isempty(use_parallel)
    use_parallel = 1;
end
if nargin < 6 || isempty(test_range)
    test_range = [0.8 1];
end

%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);
Y_train = cell2mat(cellfun(@(Y)Y(:, 1:end-1), Y_train_cell, 'UniformOutput', 0));
Y_train_diff = cell2mat(cellfun(@(Y)diff(Y, 1, 2), Y_train_cell, 'UniformOutput', 0));
Y_test = cell2mat(cellfun(@(Y)Y(:, 1:end-1), Y_test_cell, 'UniformOutput', 0));

%% MMSE prediction
Y_test_diff_hat = permute(MMSE_est(Y_train', Y_train_diff', Y_test', N_pdf, pdf_weight, memory), [3 1 2]); % Prediction of the derivative (approximated by first difference) of Y_test. The first dimension is predicted-channel, the second dimension is time, and the third dimension is predictor-channel.
Y_test_plus_hat = Y_test + Y_test_diff_hat;                                 % _plus refers to the fact that each column of Y_test_plus_hat corresponds to one time step later than the corresponding column in Y_test. This is corrected by adding a column of NaNs at the beginning to obtain Y_test_hat.

model.eq = ['$y_i(t) - y_i(t-1) = E[y_i(t) - y_i(t-1) | y_j(t-1)], i,j = 1,\dots,n$ \\ ' ...
    '``model on demand": no explicit form, $y_i(t) - y_i(t-1)$ estimated separately for any given test (query) point $y_j(t-1)$.'];

%% Output arguments
Y_test_hat = cell2mat(cellfun(@(Y)[nan(n, 1, n), Y], mat2cell(Y_test_plus_hat, n, N_test_vec-1, n), 'UniformOutput', 0));
Y_hat = [nan(n, test_ind(1), n), Y_test_hat, nan(n, N-test_ind(end), n)];   % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind), n);
end

Y_test_plus = cell2mat(cellfun(@(Y)Y(:, 2:end), Y_test_cell, 'UniformOutput', 0));
E_test = Y_test_plus_hat - Y_test_plus;                                     % Prediction error
R2 = permute(1 - sum((E_test).^2, 2) ./ sum((Y_test_plus - mean(Y_test_plus, 2)).^2, 2), [1 3 2]); % R2 is a matrix, whose (i, j) entry is the R2 of predicting y_i(t) from y_j(t-1)

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