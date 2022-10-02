function [model, R2, whiteness, Y_hat, runtime] = linear_AR(Y, include_W, n_AR_lags, W_mask, k, use_parallel, test_range)
%LINEAR_AR Fitting and cross-validating various autoregrssive linear models
% directly at the signal level.
%
%   Input arguments
% 
%   Y: a data matrix or cell array of data matrices used for system
%   identification. Each element of Y (or Y itself) is one scan, with
%   channels along the first dimension and time along the second dimension.
%   This is the only mandatory input.
% 
%   include_W: a binary flag indicating whether network iteractions should
%   be allowed in the model. Setting this to 0 only allows for self loops
%   of varying orders (scalar autoregressive models at each node). Even if
%   include_W = 1, self-loops are not allowed unless separately allowed
%   using n_AR_lags.
% 
%   n_AR_lags: number of autoregressive (AR) lags to include in the model.
%   n_AR_lags = 0 means no AR lags and in particular no self-loops.
%   n_AR_lags = 1 only allows for single-lag self loops, similar to the
%   edges allowed by include_W. n_AR_lags > 1 allows for additional AR
%   lags.
% 
%   W_mask: a character vector indicating what sparsity pattern should be
%   used for W. options are 'full', 'forward', 'backward', 'stepwise', and
%   'lasso', corresponding respectively to a dense W, a sparse W using
%   forward, backward, or stepwise regrssion, or a sparse W using LASSO
%   (1-norm) regularization. If a numeric value is provided, LASSO will be
%   selected and the numeric value will be used for the regularization
%   weight (lambda) of LASSO.
% 
%   k: number of multi-step ahead predictions for cross-validation.
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

if nargin < 2 || isempty(include_W)
    include_W = 0;
end
if nargin < 3 || isempty(n_AR_lags)
    n_AR_lags = 102;
end
AR_lags = 1:n_AR_lags;
if nargin < 4 || isempty(W_mask)
    W_mask = 1e-2;                                                     % The lambda parameter of the LASSO regularization
end
if isnumeric(W_mask)
    lambda_lasso = W_mask;
    W_mask = 'lasso';
end
if nargin < 5 || isempty(k)
    k = 1;
end
if nargin < 6 || isempty(use_parallel)
    use_parallel = 1;
end
if nargin < 7 || isempty(test_range)
    test_range = [0.8 1];
end

%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);

%% Preparing regressors
runtime_train_start = tic;

if include_W || n_AR_lags > 0
    Y_train_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, 1+n_AR_lags-lag:end-lag), AR_lags', ...
        'UniformOutput', 0)), Y_train_cell, 'UniformOutput', 0));               % A large matrix of training Y and all its time lags used for regression
    Y_train_plus = cell2mat(cellfun(@(Y)Y(:, 1+max(1, n_AR_lags):end), Y_train_cell, 'UniformOutput', 0)); % _plus regers to the time point +1. For instance, each column of Y_train_plus corresponds to one time step after the corresponding column of Y_train.
    Y_train = cell2mat(cellfun(@(Y)Y(:, max(1, n_AR_lags):end-1), Y_train_cell, 'UniformOutput', 0));
    if include_W
        Phi = [Y_train; Y_train_lags];                                          % The matrix of regressors.
        if any(strcmp(W_mask, {'forward', 'backward', 'stepwise'}))             % These methods need a 3-way breaking of the data, such that the training data is itself broken into a train-train part and a train-test one. This is due to the fact that these methods have an internal comparison between different models and that would require its own internal cross-validation.
            ind = 1:round(0.75*size(Y_train, 2));                               % This is the index of time points used for train
            ind_tst = round(0.75*size(Y_train, 2))+1:size(Y_train, 2);          % This is the index of time points used for test (other than the actual validation points put aside above)
            Y_train_tst = Y_train(:, ind_tst);                                  % The part of Y_train used for internal cross-validation, same below
            Y_train = Y_train(:, ind);
            Y_train_plus_tst = Y_train_plus(:, ind_tst);
            Y_train_plus = Y_train_plus(:, ind);
            Phi_tst = Phi(:, ind_tst);
            Phi = Phi(:, ind);
            Y_train_diff_tst = Y_train_plus_tst - Y_train_tst;
        end
    else
        Phi = Y_train_lags;
    end
    Y_train_diff = Y_train_plus - Y_train;
end

%% Least squares
Eye = logical(eye(n));
if include_W
    Theta = zeros(n, n + n*n_AR_lags);                                   % The matrix of parameters (W and D_i's if lags are included)

    switch W_mask
        case {'forward', 'stepwise'}                                        
            W_01 = false(n);                                                % A binary mask showing which entries should be initially included. For forward and stepwise regression we need a no-W version for basedline comparison below.
        case {'full', 'backward', 'lasso'}
            W_01 = ~Eye;                                                    % All edges allowed (initially).
    end

    if ~isequal(W_mask, 'lasso')
        if use_parallel
            parfor i = 1:n
                if include_W == 1
                    Ei_cell = [{Eye(:, W_01(i, :))}, repmat({Eye(:, i)}, 1, n_AR_lags)];
                else
                    if n_AR_lags == 0
                        Ei_cell = {Eye(:, W_01(i, :))};
                    else
                        Ei_cell = [{Eye(:, W_01(i, :))}, {Eye(:, i)}, ...
                            repmat({Eye(:, W_01(i, :)|Eye(i, :))}, 1, n_AR_lags-1)];
                    end
                end
                row_ind = cell2mat(cellfun(@(Ei)any(Ei, 2), Ei_cell', 'UniformOutput', 0));
                Gi = 2 * Phi(row_ind, :) * Phi(row_ind, :)';
                gi = 2 * Phi(row_ind, :) * Y_train_diff(i, :)'; 
                theta_i = zeros(n + n*n_AR_lags, 1);
                theta_i(row_ind) = lsqminnorm(Gi, gi);
                Theta(i, :) = theta_i;
            end
        else
            for i = 1:n
                if include_W == 1
                    Ei_cell = [{Eye(:, W_01(i, :))}, repmat({Eye(:, i)}, 1, n_AR_lags)];
                else
                    if n_AR_lags == 0
                        Ei_cell = {Eye(:, W_01(i, :))};
                    else
                        Ei_cell = [{Eye(:, W_01(i, :))}, {Eye(:, i)}, ...
                            repmat({Eye(:, W_01(i, :)|Eye(i, :))}, 1, n_AR_lags-1)];
                    end
                end
                row_ind = cell2mat(cellfun(@(Ei)any(Ei, 2), Ei_cell', 'UniformOutput', 0));
                Gi = 2 * Phi(row_ind, :) * Phi(row_ind, :)';
                gi = 2 * Phi(row_ind, :) * Y_train_diff(i, :)'; 
                Theta(i, row_ind) = lsqminnorm(Gi, gi);
            end
        end

        if any(strcmp(W_mask, {'forward', 'backward', 'stepwise'}))
            R2_rec = nan(n, n);
            if use_parallel
                parfor i = 1:n
                    switch W_mask
                        case 'forward'
                            [Theta(i, :), W_01(i, :), R2_rec(i, :)] = forward_selection(i, Theta(i, :), W_01(i, :), ...
                                include_W, Y_train_plus_tst, Y_train_diff_tst, Phi_tst, Phi, Y_train_diff);
                        case 'backward'
                            [Theta(i, :), W_01(i, :), R2_rec(i, :)] = backward_elimination(i, Theta(i, :), W_01(i, :), ...
                                include_W, Y_train_plus_tst, Y_train_diff_tst, Phi_tst, Phi, Y_train_diff);
                        case 'stepwise'
                            [Theta(i, :), W_01(i, :), R2_rec(i, :)] = stepwise_regression(i, Theta(i, :), W_01(i, :), ...
                                include_W, Y_train_plus_tst, Y_train_diff_tst, Phi_tst, Phi, Y_train_diff);
                    end
                end
            else
                for i = 1:n
                    switch W_mask
                        case 'forward'
                            [Theta(i, :), W_01(i, :), R2_rec(i, :)] = forward_selection(i, Theta(i, :), W_01(i, :), ...
                                include_W, Y_train_plus_tst, Y_train_diff_tst, Phi_tst, Phi, Y_train_diff);
                        case 'backward'
                            [Theta(i, :), W_01(i, :), R2_rec(i, :)] = backward_elimination(i, Theta(i, :), W_01(i, :), ...
                                include_W, Y_train_plus_tst, Y_train_diff_tst, Phi_tst, Phi, Y_train_diff);
                        case 'stepwise'
                            [Theta(i, :), W_01(i, :), R2_rec(i, :)] = stepwise_regression(i, Theta(i, :), W_01(i, :), ...
                                include_W, Y_train_plus_tst, Y_train_diff_tst, Phi_tst, Phi, Y_train_diff);
                    end
                end
            end
        end
    else
        if use_parallel
            parfor i = 1:n
                if include_W == 1
                    Ei_cell = [{Eye(:, W_01(i, :))}, repmat({Eye(:, i)}, 1, n_AR_lags)];
                else
                    if n_AR_lags == 0
                        Ei_cell = {Eye(:, W_01(i, :))};
                    else
                        Ei_cell = [{Eye(:, W_01(i, :))}, {Eye(:, i)}, ...
                            repmat({Eye(:, W_01(i, :)|Eye(i, :))}, 1, n_AR_lags-1)];
                    end
                end
                row_ind = cell2mat(cellfun(@(Ei)any(Ei, 2), Ei_cell', 'UniformOutput', 0));
                theta_i = zeros(n + n*n_AR_lags, 1);
                theta_i(row_ind) = lasso(Phi(row_ind, :)', Y_train_diff(i, :)', 'Lambda', lambda_lasso);
                Theta(i, :) = theta_i;
            end
        else
            for i = 1:n
                if include_W == 1
                    Ei_cell = [{Eye(:, W_01(i, :))}, repmat({Eye(:, i)}, 1, n_AR_lags)];
                else
                    if n_AR_lags == 0
                        Ei_cell = {Eye(:, W_01(i, :))};
                    else
                        Ei_cell = [{Eye(:, W_01(i, :))}, {Eye(:, i)}, ...
                            repmat({Eye(:, W_01(i, :)|Eye(i, :))}, 1, n_AR_lags-1)];
                    end
                end
                row_ind = cell2mat(cellfun(@(Ei)any(Ei, 2), Ei_cell', 'UniformOutput', 0));
                Theta(i, row_ind) = lasso(Phi(row_ind, :)', Y_train_diff(i, :)', 'Lambda', lambda_lasso);
            end
        end
    end
    
    if n_AR_lags == 0
        model.eq = '$y(t) - y(t-1) = W y(t-1) + e(t)$';
        model.W = Theta(:, 1:n);
    elseif  n_AR_lags == 1
        model.eq = '$y(t) - y(t-1) = W y(t-1) + e(t)$';
        model.W = Theta(:, 1:n) + Theta(:, n+1:end);
    else
        model.eq = '$y(t) - y(t-1) = W y(t-1) + \sum_{\tau\in\Tau} D_\tau y(t-\tau) + e(t)$';
        model.W = Theta(:, 1:n) + Theta(:, n+1:2*n);
        model.D = mat2cell(Theta(:, 2*n+1:end), n, n*ones(1, n_AR_lags-1));
        model.Tau = setdiff(AR_lags, 1);
    end
else
    if n_AR_lags == 0
        model.eq = '$y(t) - y(t-1) = e(t)$';
    else
        Theta = zeros(n, n*n_AR_lags);
        if use_parallel
            parfor i = 1:n
                Ei_cell = repmat({Eye(:, i)}, 1, n_AR_lags);
                row_ind = cell2mat(cellfun(@(Ei)any(Ei, 2), Ei_cell', 'UniformOutput', 0));
                Gi = 2 * Phi(row_ind, :) * Phi(row_ind, :)';
                gi = 2 * Phi(row_ind, :) * Y_train_diff(i, :)'; 
                theta_i = zeros(n*n_AR_lags, 1);
                theta_i(row_ind) = lsqminnorm(Gi, gi);
                Theta(i, :) = theta_i;
            end
        else
            for i = 1:n
                Ei_cell = repmat({Eye(:, i)}, 1, n_AR_lags);
                row_ind = cell2mat(cellfun(@(Ei)any(Ei, 2), Ei_cell', 'UniformOutput', 0));
                Gi = 2 * Phi(row_ind, :) * Phi(row_ind, :)';
                gi = 2 * Phi(row_ind, :) * Y_train_diff(i, :)'; 
                Theta(i, row_ind) = lsqminnorm(Gi, gi);
            end
        end
        model.eq = '$y(t) - y(t-1) = \sum_{\tau\in\Tau} D_\tau y(t-\tau) + e(t)$';
        model.D = mat2cell(Theta, n, n*ones(1, n_AR_lags));
        model.Tau = AR_lags;
    end
end

runtime.train = toc(runtime_train_start);

%% Cross-validated k-step ahead prediction
runtime_test_start = tic;

n_add_AR_lags = max(1, n_AR_lags) - 1;                               % This is the number of addtional zeros that need to be added to the beginning of Y_test to allow for one step ahead prediction of time points 2 and onwards.
Y_test_cell = cellfun(@(Y)[zeros(n, n_add_AR_lags), Y], Y_test_cell, 'UniformOutput', 0);
Y_test_lags = cell2mat(cellfun(@(Y)cell2mat(arrayfun(@(lag)Y(:, n_add_AR_lags+1+1-lag:end-lag), AR_lags', ...
    'UniformOutput', 0)), Y_test_cell, 'UniformOutput', 0));                % Similar to Y_train_lags but for test data. Same below.
Y_test_plus = cell2mat(cellfun(@(Y)Y(:, n_add_AR_lags+k+1:end), Y_test_cell, 'UniformOutput', 0));
Y_test = cell2mat(cellfun(@(Y)Y(:, n_add_AR_lags+1:end-1), Y_test_cell, 'UniformOutput', 0));

for i = 1:k
    if include_W
        if i == 1
            Phi = [Y_test; Y_test_lags];
        else
            Phi = [repmat(Y_test_plus_hat(:, 1:end-1), 2, 1); Phi(n+1:end-n, 1:end-1)];
        end
        Y_test_plus_hat = (eye(size(Theta)) + Theta) * Phi;
    else
        if n_AR_lags == 0
            Y_test_plus_hat = Y_test(:, 1:end-(i-1));
        else
            if i == 1
                Phi = Y_test_lags;
            else
                Phi = [Y_test_plus_hat(:, 1:end-1); Phi(1:end-n, 1:end-1)];
            end
            Y_test_plus_hat = (eye(size(Theta)) + Theta) * Phi;
        end
    end
end
Y_test_hat = cell2mat(cellfun(@(Y)[nan(n, k), Y], mat2cell(Y_test_plus_hat, n, N_test_vec-k), 'UniformOutput', 0));
Y_hat = [nan(n, test_ind(1)), Y_test_hat, nan(n, N-test_ind(end))];         % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind));
end

runtime.test = toc(runtime_test_start);
runtime.total = runtime.train + runtime.test;

E_test = Y_test_plus_hat - Y_test_plus;                                     % Prediction error
R2 = 1 - sum(E_test.^2, 2) ./ sum((Y_test_plus - mean(Y_test_plus, 2)).^2, 2);
[whiteness.p, whiteness.stat, whiteness.sig_thr] = my_whitetest_multivar(E_test);
end

%% Auxiliary Functions
function [Theta_i, W_01_i, R2_i_rec] = forward_selection(i, Theta_i, W_01_i, include_W, ...
    Y_train_plus_tst, Y_train_diff_tst, Phi_tst, Phi, Y_train_diff)              % The forward regression method. This method starts from no regressors, and adds regressors one by one until no additional benefit is obtained.
n = numel(W_01_i);                                                          
n_AR_lags = numel(Theta_i)/n-1;
Eye = logical(eye(n));
R2_i_rec = nan(1, n);
R2_i_denom = sum((Y_train_plus_tst(i, :) - mean(Y_train_plus_tst(i, :))).^2);
R2_i_rec(1) = 1 - sum((Y_train_diff_tst(i, :) - Theta_i * Phi_tst).^2, 2) ./ R2_i_denom;

W_01_i_temp = W_01_i;
Theta_i_temp = Theta_i;
i_rep = 1;
R2_diff = inf;
while R2_diff >= 0
    i_rep = i_rep + 1;
    W_01_i = W_01_i_temp;
    Theta_i = Theta_i_temp;

    R2_i_rec_rep = nan(1, n);
    Theta_i_rec = cell(1, n);
    j_vals = setdiff(find(~W_01_i), i);
    for j = j_vals
        W_01_i_temp = W_01_i | Eye(j, :);
        if include_W == 1
            Ei_cell = [{Eye(:, W_01_i_temp)}, repmat({Eye(:, i)}, 1, n_AR_lags)];
        else
            if n_AR_lags == 0
                Ei_cell = {Eye(:, W_01_i_temp)};
            else
                Ei_cell = [{Eye(:, W_01_i_temp)}, {Eye(:, i)}, ...
                    repmat({Eye(:, W_01_i_temp|Eye(i, :))}, 1, n_AR_lags-1)];
            end
        end
        row_ind = cell2mat(cellfun(@(Ei)any(Ei, 2), Ei_cell', 'UniformOutput', 0));
        Gi = 2 * Phi(row_ind, :) * Phi(row_ind, :)';
        gi = 2 * Phi(row_ind, :) * Y_train_diff(i, :)'; 
        Theta_i_rec{j} = zeros(size(Theta_i));
        Theta_i_rec{j}(row_ind) = lsqminnorm(Gi, gi);
        R2_i_rec_rep(j) = 1 - sum((Y_train_diff_tst(i, :) - Theta_i_rec{j} * Phi_tst).^2) / R2_i_denom;
    end
    [~, best_j] = max(R2_i_rec_rep);
    R2_i_rec(i_rep) = R2_i_rec_rep(best_j);
    R2_diff = R2_i_rec(i_rep) - R2_i_rec(i_rep-1);

    W_01_i_temp = W_01_i | Eye(best_j, :);                                  % This and Theta_i will only be used if the next iteration of while loop takes place
    Theta_i_temp = Theta_i_rec{best_j};
end
end

function [Theta_i, W_01_i, R2_i_rec] = backward_elimination(i, Theta_i, W_01_i, include_W, ...
    Y_train_plus_tst, Y_train_diff_tst, Phi_tst, Phi, Y_train_diff)              % The backward regression method. This method starts from all regressors and drops them one by one until no further benefit is obtained.
n = numel(W_01_i);
n_AR_lags = numel(Theta_i)/n-1;
Eye = logical(eye(n));
R2_i_rec = nan(1, n);
R2_i_denom = sum((Y_train_plus_tst(i, :) - mean(Y_train_plus_tst(i, :))).^2);
R2_i_rec(end) = 1 - sum((Y_train_diff_tst(i, :) - Theta_i * Phi_tst).^2, 2) ./ R2_i_denom;

W_01_i_temp = W_01_i;
Theta_i_temp = Theta_i;
i_rep = 1;
R2_diff = inf;
while R2_diff >= 0
    i_rep = i_rep + 1;
    W_01_i = W_01_i_temp;
    Theta_i = Theta_i_temp;

    R2_i_rec_rep = nan(1, n);
    Theta_i_rec = cell(1, n);
    j_vals = find(W_01_i);
    for j = j_vals
        W_01_i_temp = W_01_i & ~Eye(j, :);
        if include_W == 1
            Ei_cell = [{Eye(:, W_01_i_temp)}, repmat({Eye(:, i)}, 1, n_AR_lags)];
        else
            if n_AR_lags == 0
                Ei_cell = {Eye(:, W_01_i_temp)};
            else
                Ei_cell = [{Eye(:, W_01_i_temp)}, {Eye(:, i)}, ...
                    repmat({Eye(:, W_01_i_temp|Eye(i, :))}, 1, n_AR_lags-1)];
            end
        end
        row_ind = cell2mat(cellfun(@(Ei)any(Ei, 2), Ei_cell', 'UniformOutput', 0));
        Gi = 2 * Phi(row_ind, :) * Phi(row_ind, :)';
        gi = 2 * Phi(row_ind, :) * Y_train_diff(i, :)'; 
        Theta_i_rec{j} = zeros(size(Theta_i));
        Theta_i_rec{j}(row_ind) = lsqminnorm(Gi, gi);
        R2_i_rec_rep(j) = 1 - sum((Y_train_diff_tst(i, :) - Theta_i_rec{j} * Phi_tst).^2) / R2_i_denom;
    end
    [~, best_j] = max(R2_i_rec_rep);
    R2_i_rec(end-i_rep+1) = R2_i_rec_rep(best_j);
    R2_diff = R2_i_rec(end-i_rep+1) - R2_i_rec(end-i_rep+2);

    W_01_i_temp = W_01_i & ~Eye(best_j, :);                                 % This and Theta_i will only be used if the next iteration of while loop takes place
    Theta_i_temp = Theta_i_rec{best_j};
end
end

function [Theta_i, W_01_i, R2_i_rec] = stepwise_regression(i, Theta_i, W_01_i, include_W, ...
    Y_train_plus_tst, Y_train_diff_tst, Phi_tst, Phi, Y_train_diff)              % The stepwise regression method. This method is similar to forward regression except that at each step regressors may also be dropped if they are no longer beneficial (even though they once were, since new regressors have been added).
n = numel(W_01_i);
n_AR_lags = numel(Theta_i)/n-1;
Eye = logical(eye(n));
R2_i_rec = nan(1, n);
R2_i_denom = sum((Y_train_plus_tst(i, :) - mean(Y_train_plus_tst(i, :))).^2);
R2_i_rec(1) = 1 - sum((Y_train_diff_tst(i, :) - Theta_i * Phi_tst).^2, 2) ./ R2_i_denom;

W_01_i_temp = W_01_i;
Theta_i_temp = Theta_i;
i_rep = 1;
R2_diff = inf;
while R2_diff >= 0
    i_rep = i_rep + 1;
    W_01_i = W_01_i_temp;
    Theta_i = Theta_i_temp;
    [Theta_i, W_01_i, R2_i_rec_be] = backward_elimination(i, Theta_i, W_01_i, ...
        Y_train_plus_tst, Y_train_diff_tst, Phi_tst, Phi, Y_train_diff);
    R2_i_rec(i_rep-1) = max(R2_i_rec_be);

    R2_i_rec_rep = nan(1, n);
    Theta_i_rec = cell(1, n);
    j_vals = setdiff(find(~W_01_i), i);
    for j = j_vals
        W_01_i_temp = W_01_i | Eye(j, :);
        if include_W == 1
            Ei_cell = [{Eye(:, W_01_i_temp)}, repmat({Eye(:, i)}, 1, n_AR_lags)];
        else
            if n_AR_lags == 0
                Ei_cell = {Eye(:, W_01_i_temp)};
            else
                Ei_cell = [{Eye(:, W_01_i_temp)}, {Eye(:, i)}, ...
                    repmat({Eye(:, W_01_i_temp|Eye(i, :))}, 1, n_AR_lags-1)];
            end
        end
        row_ind = cell2mat(cellfun(@(Ei)any(Ei, 2), Ei_cell', 'UniformOutput', 0));
        Gi = 2 * Phi(row_ind, :) * Phi(row_ind, :)';
        gi = 2 * Phi(row_ind, :) * Y_train_diff(i, :)'; 
        Theta_i_rec{j} = zeros(size(Theta_i));
        Theta_i_rec{j}(row_ind) = lsqminnorm(Gi, gi);
        R2_i_rec_rep(j) = 1 - sum((Y_train_diff_tst(i, :) - Theta_i_rec{j} * Phi_tst).^2) / R2_i_denom;
    end
    [~, best_j] = max(R2_i_rec_rep);
    R2_i_rec(i_rep) = R2_i_rec_rep(best_j);
    R2_diff = R2_i_rec(i_rep) - R2_i_rec(i_rep-1);

    W_01_i_temp = W_01_i | Eye(best_j, :);                                  % This and Theta_i will only be used if the next iteration of while loop takes place
    Theta_i_temp = Theta_i_rec{best_j};
end
end