function [model, R2, whiteness_p, Y_hat] = nonlinear_manifold(Y, n_AR_lags, kernel, h, test_range)
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
if nargin < 5 || isempty(test_range)
    test_range = [0.8 1];
end
           
%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);

AR_lags = 1:n_AR_lags;
Y_train_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, 1+n_AR_lags-lag:end-lag), AR_lags', ...
    'UniformOutput', 0)), Y_train_cell, 'UniformOutput', 0));               % A large matrix of training Y and all its time lags used for regression
Y_train_plus = cell2mat(cellfun(@(Y)Y(:, 1+max(1, n_AR_lags):end), Y_train_cell, 'UniformOutput', 0)); % _plus regers to the time point +1. For instance, each column of Y_train_plus corresponds to one time step after the corresponding column of Y_train.
Y_train = cell2mat(cellfun(@(Y)Y(:, max(1, n_AR_lags):end-1), Y_train_cell, 'UniformOutput', 0));
Y_train_diff = Y_train_plus - Y_train;
N_train = size(Y_train_lags, 2);

n_additional_AR_lags = n_AR_lags - 1;                                       % This is the number of addtional zeros that need to be added to the beginning of Y_test to allow for one step ahead prediction of time points 2 and onwards.
Y_test_cell = cellfun(@(Y)[zeros(n, n_additional_AR_lags), Y], Y_test_cell, 'UniformOutput', 0);
Y_test_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, 1+n_AR_lags-lag:end-lag), AR_lags', ...
    'UniformOutput', 0)), Y_test_cell, 'UniformOutput', 0));                % Similar to Y_train_lags but for test data. Same below.
Y_test = cell2mat(cellfun(@(Y)Y(:, max(1, n_AR_lags):end-1), Y_test_cell, 'UniformOutput', 0));
N_test = size(Y_test_lags, 2);

%% Least squares
switch kernel
    case 'Gaussian'
        K_h = @(d)1/sqrt(2*pi)/h * exp(-d.^2/2/h^2);                        % Window function, giving the weight of each point as a function of its distance d from the testing point of interest.
    case 'Epanechnikov'
        K_h = @(d)3/4/h * max(1-d.^2/h^2, 0);
end

try
    Phi = [ones(1, N_train, N_test); Y_train_lags - permute(Y_test_lags, [1 3 2])]; % Matrix of regressors
    dist = permute(sqrt(sum(Phi(2:end, :, :).^2, 1)), [3 2 1]);                 % Pairwise Euclidean distance
    K = K_h(dist);
    low_memory = 0;
catch ME
    if any(strcmp(ME.identifier, {'MATLAB:array:SizeLimitExceeded', 'MATLAB:nomem'}))
        low_memory = 1;
    else
        throw(ME)
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
        K_i_test = K_h(dist_i_test);
        G = Phi_i_test * diag(K_i_test) * Phi_i_test';
        g = Y_train_diff * diag(K_i_test) * Phi_i_test';
    end
    theta = g * pinv(G);
    Theta0(:, i_test) = theta(:, 1);
end
                
model.eq = ['$y(t) - y(t-1) = \theta_0$ \\ ' ...
    '``model on demand": no explicit form, $\theta_0$ estimated separately for any given test (query) point.'];
model.theta_0 = Theta0;

%% Cross-validated one step ahead prediction
Y_test_plus_hat = Y_test + Theta0;                                          % _plus refers to the fact that each column of Y_test_plus_hat corresponds to one time step later than the corresponding column in Y_test. This is corrected by adding a column of NaNs at the beginning to obtain Y_test_hat.

Y_test_hat = cell2mat(cellfun(@(Y)[nan(n, 1), Y], mat2cell(Y_test_plus_hat, n, N_test_vec-1), 'UniformOutput', 0));
Y_hat = [nan(n, test_ind(1)), Y_test_hat, nan(n, N-test_ind(end))];         % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind));
end

Y_test_plus = cell2mat(cellfun(@(Y)Y(:, 1+max(1, n_AR_lags):end), Y_test_cell, 'UniformOutput', 0));
E_test = Y_test_plus_hat - Y_test_plus;                                     % Prediction error
R2 = 1 - sum(E_test.^2, 2) ./ sum((Y_test_plus - mean(Y_test_plus, 2)).^2, 2);
whiteness_p = my_whitetest(E_test');