function Y_test = LL_est(X_train, Y_train, X_test, kernel, h)
%LL_EST Locally linear estimation, based on the algorithm in
% J. Roll, \Local and piecewise affine approaches to system
% identification," Ph.D. dissertation, Linkoping University, 2003.
%
%   Y_test = LL_est(X_train, Y_train, X_test) returns a matrix (or column
%   vector) the same size as X_test containing the best prediction of the
%   corresponding output values based on a locally linear model learned
%   from (X_train, Y_train). The latter should have the same number of
%   rows, and X_train must have the same number of columns as X_test. Time
%   (observations) is along the first axis.
% 
%   Y_test = LL_est(X_train, Y_train, X_test, kernel) also determines which
%   kernel should be used for determining which training points are close
%   to each test point and how much. Options are 'Gaussian' (default) and
%   'Epanechnikov'.
% 
%   Y_test = LL_est(X_train, Y_train, X_test, kernel, h) also determines
%   the window size h. The smaller the h, the more local the predictor
%   becomes, which requires more densely sampled training data. h ->
%   infinity gives a (globally) linear predictor. 
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

if nargin < 4 || isempty(kernel)
    kernel = 'Gaussian';
end
if nargin < 5 || isempty(h)
    h = 1;
end

% Setting the distance functions that map distance d to weight K
[N_train, nx] = size(X_train);
ny = size(Y_train, 2);
N_test = size(X_test, 1);
switch kernel
    case 'Gaussian'
        K_h = @(d)1/sqrt(2*pi)/h * exp(-d.^2/2/h^2);
    case 'Epanechnikov'
        K_h = @(d)3/4/h * max(1-d.^2/h^2, 0);
end

% Computing pairwise distances
Phi = [ones(1, N_train, N_test); X_train' - permute(X_test', [1 3 2])];
dist = permute(sqrt(sum(Phi(2:end, :, :).^2, 1)), [3 2 1]);
K = K_h(dist);

% Weighted sinear regression
Theta = nan(ny, 1+nx, N_test);
for i_test = 1:N_test
    G = Phi(:, :, i_test) * diag(K(i_test, :)) * Phi(:, :, i_test)';
    g = Y_train' * diag(K(i_test, :)) * Phi(:, :, i_test)';
    Theta(:, :, i_test) = g * pinv(G);
end

% Predicting the test output values
Y_test = permute(Theta(:, 1, :), [1 3 2])';
