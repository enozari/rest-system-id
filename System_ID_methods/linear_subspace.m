function [model, R2, whiteness, Y_hat, runtime] = linear_subspace(Y, s, r, n, k, test_range)
%LINEAR_SUBSPACE Fitting and cross-validating a general linear model in
% standard state space form using subspace methods. 
% 
%   This is an implementation based on "Lennart Ljung. System
%   identification: theory for the user. PTR Prentice Hall, Upper Saddle
%   River, NJ. 1999", Chapter 10.6. See therein for detailed meaning of the
%   variables.
%
%   Input arguments
% 
%   Y: a data matrix or cell array of data matrices used for system
%   identification. Each element of Y (or Y itself) is one scan, with
%   channels along the first dimension and time along the second dimension.
%   This is the only mandatory input.
% 
%   s: number of lags of output to concatenate in order to achieve an upper
%   bound on the internal state dimension of the system.
% 
%   r: number of lags of output to concatenate to obtain the instrumental
%   variables.
% 
%   n: the dimension of the internal state space that is expected from the
%   model.
% 
%   k: number of multi-step ahead predictions for cross-validation
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

if nargin < 2 || isempty(s)
    s = 10;
end
if nargin < 3 || isempty(r)
    r = 45;
end
if nargin < 4 || isempty(n)
    n = 450;
end
if nargin < 5 || isempty(k)
    k = 1;
end
if nargin < 6 || isempty(test_range)
    test_range = [0.8 1];
end

%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, p, N] = tt_decomp(Y, test_range);

%% Estimation of the state-space matrices
runtime_train_start = tic;

n_train = numel(Y_train_cell);
G_hat_cell = cell(1, n_train);
Y_train_hankel_cell = cell(1, n_train);
Phi_cell = cell(1, n_train);
N_vec = nan(1, n_train);
for i_train = 1:n_train
    N_train_i = size(Y_train_cell{i_train}, 2);
    Y_train_i_cell = mat2cell(Y_train_cell{i_train}, p, ones(1, N_train_i));
    N_vec(i_train) = N_train_i - s - r + 1;
    hankel_ind_i = hankel(s+1:s+r, s+r:N_train_i);                          % r by N_vec(i_train)
    Y_train_hankel_cell{i_train} = cell2mat(Y_train_i_cell(hankel_ind_i));  % p*r by p*N_vec(i_train)
    toeplitz_ind_i = toeplitz(s:-1:1, s:s+N_vec(i_train)-1);                % s by N_vec(i_train)
    Phi_cell{i_train} = cell2mat(Y_train_i_cell(toeplitz_ind_i));           % p*s by p*N_vec(i_train)
    G_i = 1/N_vec(i_train) * Y_train_hankel_cell{i_train} * Phi_cell{i_train}'; % p*r by p*s
    G_hat_cell{i_train} = G_i * (N_vec(i_train) * (Phi_cell{i_train} * Phi_cell{i_train}') \ Phi_cell{i_train}); % p*r by p*N_vec(i_train)
end

G_hat = cell2mat(G_hat_cell);
[U, S, ~] = svd(G_hat);
n_max = min(size(G_hat));
if n > n_max
    warning(['Incompatible (s, r, n). Reducing n to ' num2str(n_max) '.'])
    n = n_max;
end
S_1 = S(1:n, 1:n);
U_1 = U(:, 1:n);
R = eye(n);                                                                 % Other choices include R = S_1 and R = sqrt(S_1). They did not show meaningful differences in R2 with R = I.
O_r_hat = U_1 * R;
C_hat = O_r_hat(1:p, 1:n);
A_hat = pinv(O_r_hat(1:p*(r-1), :)) * O_r_hat(p+1:end, :);

%% Estimation of covariances and recording the model
E1_hat_cell = cell(1, n_train);
E2_hat_cell = cell(1, n_train);
for i_train = 1:n_train
    X_train_i_hat = (R \ U_1') * Y_train_hankel_cell{i_train} * Phi_cell{i_train}' ... *
        * ((Phi_cell{i_train} * Phi_cell{i_train}') \ Phi_cell{i_train});
    E1_hat_cell{i_train} = X_train_i_hat(:, 2:end) - A_hat * X_train_i_hat(:, 1:end-1);
    E2_hat_cell{i_train} = Y_train_cell{i_train}(:, s+1:s+N_vec(i_train)-1) - C_hat * X_train_i_hat(:, 1:end-1);
end

E1_hat = cell2mat(E1_hat_cell);
E2_hat = cell2mat(E2_hat_cell);
full_cov = cov([E1_hat' E2_hat']);
Qcov = full_cov(1:n, 1:n);
Mcov = full_cov(1:n, n+1:end);
Rcov = full_cov(n+1:end, n+1:end);

model.eq = ['$$x(t) - x(t-1) = W x(t-1) + e_1(t) \\ ' ...
    'y(t) = C x(t) + e_2(t) \\ ' ...
    'Cov([e_1(t); e_2(t)]) = [Q M; M^T R]$$'];
model.W = A_hat - eye(n);
model.C = C_hat;
model.Q = Qcov;
model.M = Mcov;
model.R = Rcov;

runtime.train = toc(runtime_train_start);

%% Cross-validated k-step ahead prediction. At any time t, this is achieved via a standard Kalman predictor using output data from the first time point until time t-1.
runtime_test_start = tic;

n_test = numel(Y_test_cell);
Y_test_hat = cell(1, n_test);                                               % The prediction of Y_test_cell
A_hat_mod = A_hat - Mcov * (Rcov \ C_hat);
Rcov_mod = Mcov * (Rcov \ Mcov');
for i_test = 1:n_test
    Y_test = Y_test_cell{i_test};
    N_test = N_test_vec(i_test);
    X_test_hat_pred_1 = nan(n, N_test);                                     % Predicted state
    X_test_hat = nan(n, N_test);                                            % Filtered state
    X_test_hat(:, 1) = 0;
    P_hat_pred = nan(n, n, N_test);                                         % Predicted state covariance
    P_hat = nan(n, n, N_test);                                              % Filtered state covariance
    P_hat(:, :, 1) = eye(n);
    for t = 1:N_test-1
        X_test_hat_pred_1(:, t+1) = A_hat_mod * X_test_hat(:, t) + Mcov * (Rcov \ Y_test(:, t));
        P_hat_pred(:, :, t+1) = A_hat_mod * P_hat(:, :, t) * A_hat_mod' + Qcov - Rcov_mod;
        X_test_hat(:, t+1) = X_test_hat_pred_1(:, t+1) + P_hat_pred(:, :, t+1) * C_hat' ... *
            * ((C_hat * P_hat_pred(:, :, t+1) * C_hat' + Rcov) \ (Y_test(:, t+1) - C_hat * X_test_hat_pred_1(:, t+1)));
        P_hat(:, :, t+1) = P_hat_pred(:, :, t+1) - P_hat_pred(:, :, t+1) * C_hat' ... *
            * ((C_hat * P_hat_pred(:, :, t+1) * C_hat' + Rcov) \ (C_hat * P_hat_pred(:, :, t+1)));
    end
    Y_test_hat_i_test_k = cell(k, 1);
    Y_test_hat_i_test_k{1} = C_hat * X_test_hat_pred_1;
    for i = 2:k
        X_test_hat_pred_i = nan(n, N_test);
        for t = i+1:N_test
            X_test_hat_pred_i(:, t) = X_test_hat_pred_1(:, t-i+1);
            for j = 2:i
                X_test_hat_pred_i(:, t) = A_hat_mod * X_test_hat_pred_i(:, t) ... +
                    + Mcov * (Rcov \ Y_test_hat_i_test_k{j-1}(:, t-i+j));   % Uncorrected (open-loop) predictions starting from the 1-step prediction at time t-i+1.
            end
        end
        Y_test_hat_i_test_k{i} = C_hat * X_test_hat_pred_i;
    end
    Y_test_hat{i_test} = Y_test_hat_i_test_k{k};
end

Y_test_full = cell2mat(Y_test_cell);
Y_test_hat_full = cell2mat(Y_test_hat);

Y_hat = [nan(p, test_ind(1)), Y_test_hat_full, nan(p, N-test_ind(end))];     % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, p, diff(break_ind));
end

runtime.test = toc(runtime_test_start);
runtime.total = runtime.train + runtime.test;

nan_ind = any(isnan(Y_test_hat_full));
E2_test = Y_test_hat_full(:, ~nan_ind) - Y_test_full(:, ~nan_ind);          % Prediction error
R2 = 1 - sum(E2_test.^2, 2) ./ sum((Y_test_full(:, ~nan_ind) - mean(Y_test_full(:, ~nan_ind), 2)).^2, 2);
[whiteness.p, whiteness.stat, whiteness.sig_thr] = my_whitetest_multivar(E2_test);