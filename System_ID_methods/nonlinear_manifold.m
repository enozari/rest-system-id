function [model, R2, whiteness, Y_hat, runtime] = nonlinear_manifold(Y, n_AR_lags, kernel, h, ...
    scale_h, k, test_range)
%NONLINEAR_MANIFOLD Fitting and cross-validating a nonlinear model using
% locally linear approximation of its vector field
%
%   Input arguments
% 
%   Y: a data matrix or cell array of data matrices used for system
%   identification. Each element of Y (or Y itself) is one scan, with
%   channels along the first dimension and time along the second dimension.
%   This is the only mandatory input.
% 
%   kernel: the kernel to use for weighting training points based on their
%   distance to each test point (a.k.a. the window function). Options are
%   'Gaussian' and 'Epanechnikov'.
% 
%   h: the size of the window mentioned above. The smaller the h, the more
%   local the model is. h -> infinity makes the model equivalent to a
%   (globally) linear model.
% 
%   scale_h: struct determining whether and how to scale the window size h.
%   the filed scale_h.do_scale is mandatory and is a binary flag indicating
%   whether h should be re_scaled using the data at hand. If true, then a
%   second field scale_h.base_med_dist is also required, including the
%   median distance between all pairs of training and test samples in the
%   original data set over which the value of h has been selected. The
%   algorithm then adjusts h using the median distance between all pairs of
%   training and test samples in the current data (Y).
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

if nargin < 2 || isempty(n_AR_lags)
    n_AR_lags = 3;
end
if nargin < 3 || isempty(kernel)
    kernel = 'Gaussian';
end
if isnumeric(kernel)
    switch kernel
        case 1
            kernel = 'Gaussian';
        case 2
            kernel = 'Epanechnikov';
        otherwise
            error('Kernel can be either ''Gaussian'' (1) or ''Epanechnikov'' (2)');
    end
end
if nargin < 4 || isempty(h)
    h = 1e4;
end
if nargin < 5 || isempty(scale_h)
    scale_h.do_scale = false;
end
if nargin < 6 || isempty(k)
    k = 1;
end
if nargin < 7 || isempty(test_range)
    test_range = [0.8 1];
end
           
%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);

model.eq = ['$y(t) - y(t-1) = \theta_0$ \\ ' ...
    '``model on demand": no explicit form, $\theta_0$ estimated separately for any given test (query) point $(y_i(t-1), y_i(t-2), ..., y_i(t-d))$.'];
model.d = n_AR_lags;

runtime.train = 0;

%% Cross-validated k-step ahead prediction
runtime_test_start = tic;

AR_lags = 1:n_AR_lags;
Y_train_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, 1+n_AR_lags-lag:end-lag), AR_lags', ...
    'UniformOutput', 0)), Y_train_cell, 'UniformOutput', 0));               % A large matrix of training Y and all its time lags used for regression
Y_train_plus = cell2mat(cellfun(@(Y)Y(:, 1+max(1, n_AR_lags):end), Y_train_cell, 'UniformOutput', 0)); % _plus regers to the time point +1. For instance, each column of Y_train_plus corresponds to one time step after the corresponding column of Y_train.
Y_train = cell2mat(cellfun(@(Y)Y(:, max(1, n_AR_lags):end-1), Y_train_cell, 'UniformOutput', 0));
Y_train_diff = Y_train_plus - Y_train;
N_train = size(Y_train_lags, 2);

n_add_AR_lags = n_AR_lags - 1;                                       % This is the number of addtional zeros that need to be added to the beginning of Y_test to allow for one step ahead prediction of time points 2 and onwards.
Y_test_cell = cellfun(@(Y)[zeros(n, n_add_AR_lags), Y], Y_test_cell, 'UniformOutput', 0);
Y_test_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, n_add_AR_lags+1+1-lag:end-lag), AR_lags', ...
    'UniformOutput', 0)), Y_test_cell, 'UniformOutput', 0));                % Similar to Y_train_lags but for test data. Same below.

switch kernel
    case 'Gaussian'
        K_h = @(d, h)1/sqrt(2*pi)/h * exp(-d.^2/2/h^2);                        % Window function, giving the weight of each point as a function of its distance d from the testing point of interest.
    case 'Epanechnikov'
        K_h = @(d, h)3/4/h * max(1-d.^2/h^2, 0);
end

if scale_h.do_scale
    h_orig = h;
end

low_memory = 0;
for i = 1:k
    N_test = size(Y_test_lags, 2);
    if ~low_memory
        try
            Phi = [ones(1, N_train, N_test); Y_train_lags - permute(Y_test_lags, [1 3 2])]; % Matrix of regressors
            dist = permute(sqrt(sum(Phi(2:end, :, :).^2, 1)), [3 2 1]);                 % Pairwise Euclidean distance
            if scale_h.do_scale
                h = h_orig * median(dist(:)) / scale_h.base_med_dist;
            end
            K = K_h(dist, h);
            low_memory = 0;
        catch ME
            if any(strcmp(ME.identifier, {'MATLAB:array:SizeLimitExceeded', 'MATLAB:nomem'}))
                low_memory = 1;
            else
                throw(ME)
            end
        end
    end

    Theta0 = nan(n, N_test);                                                    % The matrix of parameters. Each slice is for one test point. Within each slice, the first column is the zero'th order Taylor term (at the corresponding test point) and the remaining n x n matrix is the coefficient of the linear Taylor term (which is discarded).
    for i_test = 1:N_test
        if ~low_memory
            G = Phi(:, :, i_test) * diag(K(i_test, :)) * Phi(:, :, i_test)';
            g = Y_train_diff * diag(K(i_test, :)) * Phi(:, :, i_test)';
        else
            Phi_i_test = [ones(1, N_train); Y_train_lags - Y_test_lags(:, i_test)]; % Matrix of regressors
            dist_i_test = sqrt(sum(Phi_i_test(2:end, :).^2, 1));                              % Pairwise Euclidean distance
            if scale_h.do_scale
                h = h_orig * median(dist_i_test) / scale_h.base_med_dist;
            end
            K_i_test = K_h(dist_i_test, h);
            G = Phi_i_test * diag(K_i_test) * Phi_i_test';
            g = Y_train_diff * diag(K_i_test) * Phi_i_test';
        end
        theta = g * pinv(G);
        Theta0(:, i_test) = theta(:, 1);
    end
    Y_test_plus_hat = Y_test_lags(1:n, :) + Theta0;                                          % _plus refers to the fact that each column of Y_test_plus_hat corresponds to one time step later than the corresponding column in Y_test. This is corrected by adding a column of NaNs at the beginning to obtain Y_test_hat.
    
    Y_test_lags = [Y_test_plus_hat(:, 1:end-1); Y_test_lags(1:end-n, 1:end-1)];
end

Y_test_hat = cell2mat(cellfun(@(Y)[nan(n, k), Y], mat2cell(Y_test_plus_hat, n, N_test_vec-k), 'UniformOutput', 0));
Y_hat = [nan(n, test_ind(1)), Y_test_hat, nan(n, N-test_ind(end))];         % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind));
end

runtime.test = toc(runtime_test_start);
runtime.total = runtime.train + runtime.test;

Y_test_plus = cell2mat(cellfun(@(Y)Y(:, n_add_AR_lags+k+1:end), Y_test_cell, 'UniformOutput', 0));
E_test = Y_test_plus_hat - Y_test_plus;                                     % Prediction error
R2 = 1 - sum(E_test.^2, 2) ./ sum((Y_test_plus - mean(Y_test_plus, 2)).^2, 2);
[whiteness.p, whiteness.stat, whiteness.sig_thr] = my_whitetest_multivar(E_test);