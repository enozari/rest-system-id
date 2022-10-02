function [model, R2, whiteness, Y_hat, runtime] = linear_neural(Y, TR, n_h, n_phi, n_psi, W_mask, k, ...
    use_parallel, test_range)
%LINEAR_NEURAL Fitting and cross-validating a general family of linear
% models that include a model for the hemodynamic response function (HRF)
% (and therefore have states at the "neural" level).
%
%   Input arguments
% 
%   Y: a data matrix or cell array of data matrices used for system
%   identification. Each element of Y (or Y itself) is one scan, with
%   channels along the first dimension and time along the second dimension.
%   This is the only mandatory input.
% 
%   TR: Sampling time. 
% 
%   n_h, n_phi, n_psi: integer, degree of finite impulse response (FIR)
%   approximations of the HRF, process noise model, and observation noise
%   model respectively.
% 
%   n_EM_iter: integer, number of iterations between parameter estimation
%   and state estimation.
% 
%   alpha: scalar in [0, 1] indicating the weight of fitting the process
%   model (state equation) relative to fitting the observation model
%   (output equation).
% 
%   W_mask: a character vector indicating what sparsity pattern should be
%   used for W. options are 'full' and 'lasso', corresponding respectively
%   to a dense W and a sparse W using LASSO (1-norm) regularization.
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

if nargin < 2 || isempty(TR)
    TR = 0.72;
end
if nargin < 3 || isempty(n_h)
    n_h = 5;
end
if nargin < 4 || isempty(n_phi)
    n_phi = 5;
end
if nargin < 5 || isempty(n_psi)
    n_psi = 5;
end
if nargin < 6 || isempty(W_mask)
    W_mask = 1e-2;                                                     % The lambda parameter of the LASSO regularization
end
if nargin < 7 || isempty(k)
    k = 1;
end
if nargin < 8 || isempty(use_parallel)
    use_parallel = 1;
end
if nargin < 9 || isempty(test_range)
    test_range = [0.8 1];
end

deconv_method = 'none';                                                     % Method of deconvolution for obtaining the state time series for the first round of iterating between parameter and state estimation. Options are 'none', where state is initially taken to be the output, or 'Wiener' where Wiener deconvlution is applied to the BOLD time series.
if isequal(deconv_method, 'Wiener')
    HRF_deconv_SNR = 0.5;                                                   % SNR estimate required for the Wiener deconvolution method.
end
n_EM_iter = 2;
max_lag = max(n_phi+1, n_h+n_psi);                                          % Maximum number of lags of state induced due to the FIR filters.
num_output_lags = max_lag * 3;                                              % In computing the state estimate in cross-validation, using the entire history of states from time 0 to time t is not necessary if t is large but results in very large computational complexity. This variable indicates the number of lags effectively needed. 3~4 times max_lag was enough to be enough using trial and error.

%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);

runtime_train_start = tic;

Y_train_lags = cell2mat(cellfun(@(Y)extract_lags(Y, 0:n_psi, max_lag), Y_train_cell, 'UniformOutput', 0)); % A large matrix of training Y and all its time lags used for regression

%% Obtaining initial X_train_lags
switch deconv_method
    case 'Wiener'
        X_train_cell = cellfun(@(Y)HRF_deconv(Y, TR, HRF_deconv_SNR), Y_train_cell, 'UniformOutput', 0); % Cell array the same size as Y_train_cell but for internal states X
    case 'none'
        X_train_cell = Y_train_cell;
end
X_train_lags = cell2mat(cellfun(@(X)extract_lags(X, 0:max_lag, max_lag), X_train_cell, 'UniformOutput', 0)); % Similar to Y_train_lags but for states

%% Parameter (and state) estimation
W_rec = nan(n, n, n_EM_iter);                                               % Each n x n slice of W_rec is the effective connectivity matrix at one iteration of the algorithm. 
H_rec = nan(n, n_h, n_EM_iter);                                             % H_rec(i, :, i_EM_iter) is the coefficients of the FIR approximation of the HRF of region i at iteration i_EM_iter. The first coefficient starts from lag 1. Similarly for Phi_rec and Psi_rec below. See model.eq below for more details.
Phi_rec = nan(n, n_phi, n_EM_iter);                                         
Psi_rec = nan(n, n_psi, n_EM_iter);

h_0 = nominal_HRF_coeffs(TR, n_h);                                          % FIR coefficients of a nominal HRF as an initialization of the parameters in H_rec.

Phi = zeros(n, n_phi);
Psi = zeros(n, n_psi);
for i_EM_iter = 1:n_EM_iter                                                 % Iterating between parameter estimation and state estimation n_EM_iter times
    [W, H, Phi, Psi] = PE_id(Y_train_lags, X_train_lags, W_mask, n_h, Phi, Psi, h_0); % The parameter estimation step
    W_rec(:, :, i_EM_iter) = W;
    H_rec(:, :, i_EM_iter) = H;
    Phi_rec(:, :, i_EM_iter) = Phi;
    Psi_rec(:, :, i_EM_iter) = Psi;

    X_train_lags = cell2mat(cellfun(@(Y)state_est(Y, W, H, Phi, Psi), Y_train_cell, 'UniformOutput', 0)); % The state estimation step
end

model.eq = ['$$x(t) - x(t-1) = W x(t-1) + G_1(q) e_1(t) \\ ' ...
    'y(t) = H(q) x(t) + G_2(q) e_2(t) \\ ' ...
    'H(q) = \sum_{p=1}^{n_h} \diag(H_{:,p}) q^{-p} \\ ' ...
    'F_1(q) = I - G_1^{-1}(q) = \sum_{p=1}^{n_\phi} \diag(\Phi_{:,p}) q^{-p} \\ ' ...
    'F_2(q) = I - G_2^{-1}(q) = \sum_{p=1}^{n_\psi} \diag(\Psi_{:,p}) q^{-p}$$'];
model.n_h = n_h;
model.n_phi = n_phi;
model.n_psi = n_psi;

runtime.train = toc(runtime_train_start);

%% Cross-validated k-step ahead prediction
runtime_test_start = tic;

n_test = numel(Y_test_cell);                                                % Number of distinct test data segments
Y_test_full = cell2mat(Y_test_cell);

Y_test_hat_rec = arrayfun(@(N_test)nan(n, N_test, n_EM_iter), N_test_vec, 'UniformOutput', 0); % A cell array the same size as Y_test_cell for predicted Y
R2_rec = nan(n, n_EM_iter);
E_test_rec = cell(1, 1, n_EM_iter);                                         % Prediction error
nan_ind = arrayfun(@(N_test)nan(1, N_test, n_EM_iter), N_test_vec, 'UniformOutput', 0); % Index of time points at which a cross-validated prediction is not available, either because they correspond to training samples, or they are the first sample of their recording, or the state estimator returned NaN because of ill-conditioned matrices
for i_EM_iter = 1:n_EM_iter
    W = W_rec(:, :, i_EM_iter);
    H = H_rec(:, :, i_EM_iter);
    Phi = Phi_rec(:, :, i_EM_iter);
    Psi = Psi_rec(:, :, i_EM_iter);
    
    Delta = extract_Delta(H, Psi);                                          % The difference between H and the convolution of Psi and H, see extract_Delta below. This filter naturally arises in the output equation.
    [~, A_T] = extract_sprad(W, Phi, max_lag);                              % The first block-row of the adjacency matrix of the system if written in standard state space form. Note that by standard state space form, we mean after concatenating X and all its lags into a large state vector whose dynamics become first order. See extract_sprad functin below.
    
    for i_test = 1:n_test
        Y_test = [zeros(n, 2*max_lag), Y_test_cell{i_test}];                % Padding Y_test_cell{i_test} with enough zero columns such that its prediction can be done from its second time point (the first time point is never predictable by definition).
        Y_test_lags = extract_lags(Y_test, 1:n_psi, max_lag);
        Y_test_lags(:, 1:max_lag+1, :) = [];
        N_test = N_test_vec(i_test) + 2*max_lag;                            % Taking into account the 2*max_lag zero columns that we added to the beginning of Y_test_cell entires
        X_test_lags = nan(n, N_test, max_lag+1);                            % Estimate of the state corresponding to Y_test_lags
        
        Y_test_hat_k = cell(k-1, 1);
        for i = 1:k
            if use_parallel
                parfor t = max_lag+1:N_test                                     % For each test point t, we estimate the state using outputs from beginning all the way to time t-1. If t is large, outputs very far in the past are not informative, and we only go num_output_lags steps into the past.
                    warning('off')
                    Y_test4state_est = Y_test(:, max(1, t-num_output_lags+1):t-(i-1));
                    for j = 1:i-1
                        Y_test4state_est = [Y_test4state_est, Y_test_hat_k{j}(:, t-(i-1)+j)];
                    end
                    X_test_lags_t = state_est(Y_test4state_est, W, H, Phi, Psi);
                    warning('on')
                    X_test_lags(:, t, :) = X_test_lags_t(:, end, :);
                end
            else
                for t = max_lag+1:N_test
                    warning('off')
                    Y_test4state_est = Y_test(:, max(1, t-num_output_lags+1):t-(i-1));
                    for j = 1:i-1
                        Y_test4state_est = [Y_test4state_est, Y_test_hat_k{j}(:, t-(i-1)+j)];
                    end
                    X_test_lags_t = state_est(Y_test4state_est, W, H, Phi, Psi);
                    warning('on')
                    X_test_lags(:, t, :) = X_test_lags_t(:, end, :);
                end
            end
            X_test_lags(:, 1:max_lag, :) = [];                                  % Removing max_lag time points at the beginning since they are not all in theory predictable from Y_test. Recall that 2*max_lag zeros were added above, so still we have max_lag entries extra.

            X_test_lags_2D = cell2mat(reshape(mat2cell(X_test_lags, n, size(X_test_lags, 2), ones(1, size(X_test_lags, 3))), [], 1)); % Concatenating the 3D layers of X_test_lags along the first dimension to prepare it for matrix multiplication.
            X_test_hat = A_T * X_test_lags_2D(n+1:end, 1:end-1);                % Running the state equation one step ahead (state one step ahead prediction)
            X_test_hat_lags = extract_lags(X_test_hat, 1:n_h+n_psi, max_lag);     % Extrating lags since lags were lost in the forward simulation above. This step strips max_lag more time points from the beginning, so X_test_hat_lags has the same number of time points as Y_test_cell entries had originally.
            Y_test_hat = sum(permute(Psi, [1 3 2]) .* Y_test_lags, 3) ... +
                + sum(permute(Delta, [1 3 2]) .* X_test_hat_lags, 3); % Running the output equation (computing predicted output from predicted state)
            Y_test_hat = [nan(n, 1), Y_test_hat];                               % The first time point of each test segment cannot be predicted by definition in one step ahead prediction
            
            Y_test_hat_k{i} = [zeros(n, 2*max_lag+i), Y_test_hat(:, i+1:end)];
            Y_test_lags = cat(3, Y_test_hat(:, 1:end-1), ...
                [nan(n, 1, n_psi-1), Y_test_lags(:, 1:end-1, 1:end-1)]);
        end
        
        Y_test_hat_rec{i_test}(:, :, i_EM_iter) = Y_test_hat;               % Book keeping

        nan_ind{i_test}(:, :, i_EM_iter) = any(isnan(Y_test_hat));          % In addition to the first time point, some first few columns of Y_test_hat may contain NANs inheried from the first few slice-columns of X_test_lags because of numerical instabilities in state estimation with very few time points.
        if nnz(nan_ind{i_test}(:, :, i_EM_iter)) > 0.05 * N_test
            warning('Too many NAN columns in Y_test_hat. Check the state estimation and X_test_lags or increase the number of test points.')
        end
    end
        
    nan_ind_full = cell2mat(cellfun(@(ind)ind(:, :, i_EM_iter), nan_ind, 'UniformOutput', 0)); % Concatenating the i_EM_iter'th slice of nan_ind. Same below.
    Y_test_hat_full = cell2mat(cellfun(@(Y)Y(:, :, i_EM_iter), Y_test_hat_rec, 'UniformOutput', 0));
    E_test_rec{i_EM_iter} = Y_test_hat_full(:, ~nan_ind_full) - Y_test_full(:, ~nan_ind_full);
    R2_rec(:, i_EM_iter) = 1 - sum(E_test_rec{i_EM_iter}.^2, 2) ... ./
        ./ sum((Y_test_full(:, ~nan_ind_full) - mean(Y_test_full(:, ~nan_ind_full), 2)).^2, 2);
end

R2_cmp = median(R2_rec)' - median(R2_rec);
best_EM_iter = find(all(R2_cmp >= 0, 2));                                   % Finding the iteration with the best median R^2

model.W = W_rec(:, :, best_EM_iter);
model.H = H_rec(:, :, best_EM_iter);
model.Phi = Phi_rec(:, :, best_EM_iter);
model.Psi = Psi_rec(:, :, best_EM_iter);

Y_test_hat_full = cell2mat(cellfun(@(Y)Y(:, :, best_EM_iter), Y_test_hat_rec, 'UniformOutput', 0));
Y_hat = [nan(n, test_ind(1)), Y_test_hat_full, nan(n, N-test_ind(end))];
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind));
end

runtime.test = toc(runtime_test_start);
runtime.total = runtime.train + runtime.test;

R2 = R2_rec(:, best_EM_iter);
[whiteness.p, whiteness.stat, whiteness.sig_thr] = my_whitetest_multivar(E_test_rec{best_EM_iter});
end

%% Auxiliary functions
function Z_lags = extract_lags(Z, lags, max_lag)                            % Function to extract max_lags lags of a vector time series Z and placing them on top of it along the third dimension
Z_lags = cell2mat(arrayfun(@(lag)Z(:, max_lag-lag+1:end-lag), permute(lags, [1 3 2]), 'UniformOutput', 0));
end

function [W, H, Phi, Psi] = PE_id(Y_lags, X_lags, W_mask, n_h, Phi, Psi, h_0) % Function for parameter estimation. The state is assumed to be known here.
if isnumeric(W_mask)
    lambda_lasso = W_mask;                                                  % The lambda parameter in LASSO regularization, only used if W_mask = 'lasso'.
    W_mask = 'lasso';
end

n = size(Y_lags, 1);        
n_phi = size(Phi, 2);
n_psi = size(Psi, 2);
W = nan(n);
H = nan(n, n_h);

lambda_h = 1e7;                                                             % Regularization constant for the least squares corresponding to H
PE_id_stop_ratio = 0.01;                                                    % Stopping criteria for the coordinate descent in the identification step. A minimum of 10 coordinate ascent iterations are always performed regardless of this stopping creteria.

max_iter = 100;                                                             % Maximum number of coordinate ascent iterations between estimation of W and Phi, or H and Psi. Note that the optimization is quadratic in W when Phi is fixed and vice versa (same for H and Psi).
W_rec = nan(n, n, max_iter);
H_rec = nan(n, n_h, max_iter);
Phi_rec = cat(3, Phi, nan(n, n_phi, max_iter-1));
Psi_rec = cat(3, Psi, nan(n, n_psi, max_iter-1));

% Identifying W & Phi
for i = 1:n
    iter = 2;
    large_error = 1;                                                        % Flag indicating that the coordinate ascent hasn't converged yet.
    while iter <= max_iter && large_error
        % Identifying W for fixed Phi (linear regression)
        phi_i_3D = permute(Phi(i, :), [1 3 2]);
        G_i = X_lags(:, :, 2) - sum(X_lags(:, :, 3:n_phi+2) .* phi_i_3D, 3);
        C_i = X_lags(i, :, 2) - X_lags(i, :, 1) ... +
            + sum((X_lags(i, :, 2:n_phi+1) - X_lags(i, :, 3:n_phi+2)) .* phi_i_3D, 3);
        if isequal(W_mask, 'full')
            Gamma_i = G_i * G_i';
            gamma_i = G_i * C_i';
            W(i, :) = lsqminnorm(Gamma_i, -gamma_i);
        elseif isequal(W_mask, 'lasso')
            W(i, :) = lasso(G_i', -C_i', 'Lambda', lambda_lasso);
        else
            error('Unsupported W_mask')
        end
        W_rec(i, :, iter) = W(i, :);
        
        % Identifying Phi for fixed W (linear regression)
        G_i = permute(X_lags(i, :, 2:n_phi+1) - X_lags(i, :, 3:n_phi+2) ... -
            - sum(X_lags(:, :, 3:n_phi+2) .* W(i, :)', 1), [3 2 1]);
        C_i = X_lags(i, :, 2) - X_lags(i, :, 1) + W(i, :) * X_lags(:, :, 2);
        Gamma_i = G_i * G_i';
        gamma_i = G_i * C_i';
        Phi(i, :) = lsqminnorm(Gamma_i, -gamma_i);
        Phi_rec(i, :, iter) = Phi(i, :);
        
        % Book-keeping
        if iter < 10
            large_error = 1;
        else
            W_rel_error = norm(W_rec(i, :, iter) - W_rec(i, :, iter-1)) / norm(W_rec(i, :, iter-1));
            Phi_rel_error = norm(Phi_rec(i, :, iter) - Phi_rec(i, :, iter-1)) / norm(Phi_rec(i, :, iter-1));
            large_error = max(W_rel_error, Phi_rel_error) > PE_id_stop_ratio;
        end
        iter = iter + 1;
    end
end

% Identifying H & Psi
for i = 1:n
    X_i_bar = cell2mat(arrayfun(@(t)permute(hankel(X_lags(i, t, 3:n_psi+2), X_lags(i, t, n_psi+2:n_h+n_psi+1)), ...
        [1 3 2]), 1:size(X_lags, 2), 'UniformOutput', 0));                  % A 3D array of Hankel matrices useful for writing least squares in closed form.
    iter = 2;
    large_error = 1;                                                        % Flag indicating that the coordinate ascent hasn't converged yet.
    while iter <= max_iter && large_error
        % Identifying H for fixed Psi (linear regression)
        G_i = permute(X_lags(i, :, 2:n_h+1) - sum(X_i_bar .* Psi(i, :)', 1), [3 2 1]);
        C_i = sum(Y_lags(i, :, 2:n_psi+1) .* permute(Psi(i, :), [1 3 2]), 3) - Y_lags(i, :, 1);
        Gamma_i = G_i * G_i';
        gamma_i = G_i * C_i';
        H(i, :) = lsqminnorm(Gamma_i + lambda_h*eye(n_h), -gamma_i + lambda_h*h_0);
        H_rec(i, :, iter) = H(i, :);
        
        % Identifying Psi for fixed H (linear regression)
        h_i_3D = permute(H(i, :), [1 3 2]);
        G_i = permute(Y_lags(i, :, 2:n_psi+1), [3 2 1]) - sum(X_i_bar .* h_i_3D, 3);
        C_i = sum(X_lags(i, :, 2:n_h+1) .* h_i_3D, 3) - Y_lags(i, :, 1);
        Gamma_i = G_i * G_i';
        gamma_i = G_i * C_i';
        Psi(i, :) = lsqminnorm(Gamma_i, -gamma_i);
        Psi_rec(i, :, iter) = Psi(i, :);
        
        % Book-keeping
        if iter < 10
            large_error = 1;
        else
            H_rel_error = norm(H_rec(i, :, iter) - H_rec(i, :, iter-1)) / norm(H_rec(i, :, iter-1));
            Psi_rel_error = norm(Psi_rec(i, :, iter) - Psi_rec(i, :, iter-1)) / norm(Psi_rec(i, :, iter-1));
            large_error = max(H_rel_error, Psi_rel_error) > PE_id_stop_ratio;
        end
        
        iter = iter + 1;
    end
end
end

function X_lags = state_est(Y, W, H, Phi, Psi)                       % Function for state estimation. The model parameters are assumed to be known. This is equivalent to Kalman filtering but is not recursive since the Ricatti iteration of Kalman filter diverges at the very large dimension and close to unstable (if not fully unstable) condition of our system.
N = size(Y, 2);
n = size(W, 1);
n_h = size(H, 2);
n_phi = size(Phi, 2);
n_psi = size(Psi, 2);
max_lag = max(n_phi+1, n_h+n_psi);
alpha = 0.1;

D = nan(n, n, n_phi+2);
D(:, :, 1) = -eye(n);
D(:, :, 2) = eye(n) + W + diag(Phi(:, 1));
for p = 2:n_phi
    D(:, :, p+1) = diag(Phi(:, p) - Phi(:, p-1)) - Phi(:, p-1) .* W;
end
D(:, :, n_phi+2) = -diag(Phi(:, n_phi)) - Phi(:, n_phi) .* W;

D_flipped = flip(D, 3);
[p_mat, q_mat] = meshgrid(1:n_phi+2);                                       % q is the row index, p is the column index
triu_ind = find(triu(true(n_phi+2)));
Gamma1_core_triu = arrayfun(@(q, p)D_flipped(:, :, q)' * D_flipped(:, :, p), q_mat(triu_ind), p_mat(triu_ind), 'UniformOutput', 0);
Gamma1_core = cell(n_phi+2);
Gamma1_core(triu_ind) = Gamma1_core_triu;
Gamma1_core_transpose = cellfun(@transpose, Gamma1_core, 'UniformOutput', 0)';
strict_tril_ind = find(tril(true(n_phi+2), -1));
Gamma1_core(strict_tril_ind) = Gamma1_core_transpose(strict_tril_ind);
Gamma1_core = cell2mat(Gamma1_core);

Gamma1 = spalloc(n*N, n*N, (n_phi+2)^2*n^2+(N-max_lag-1)*(2*n_phi+3)*n^2);
for t = 1:N-max_lag
    block_inds = t+max_lag-n_phi-1:t+max_lag;
    inds = (block_inds(1)-1)*n+1:block_inds(end)*n;
    Gamma1(inds, inds) = Gamma1(inds, inds) + Gamma1_core;
end

Delta = extract_Delta(H, Psi);

Delta_flipped = fliplr(Delta);
Gamma2_core = cell2mat(cellfun(@(cll)sparse(diag(cll)), ...
    mat2cell(reshape(Delta_flipped .* permute(Delta_flipped, [1 3 2]), n*(n_h+n_psi), n_h+n_psi), ...
    n*ones(1, n_h+n_psi), ones(1, n_h+n_psi)), 'UniformOutput', 0));

Gamma2 = spalloc(n*N, n*N, (n_h+n_psi)^2*n+(N-max_lag-1)*(2*n_h+2*n_psi-1)*n);
for t = 1:N-max_lag
    block_inds = t+max_lag-n_h-n_psi:t+max_lag-1;
    inds = (block_inds(1)-1)*n+1:block_inds(end)*n;
    Gamma2(inds, inds) = Gamma2(inds, inds) + Gamma2_core;
end

Y_lags = extract_lags(Y, 0:n_psi, max_lag);
Y_S = sum(permute(Psi, [1 3 2]) .* Y_lags(:, :, 2:end), 3) - Y_lags(:, :, 1);
gamma2 = reshape(sum(cell2mat(arrayfun(@(p)[zeros(n, max_lag-p) Delta(:, p).*Y_S zeros(n, p)], ...
    permute(1:n_h+n_psi, [1 3 2]), 'UniformOutput', 0)), 3), [], 1);

Gamma1 = (alpha/(1-alpha))*Gamma1;                                          % The following 3 lines is a memory efficient way of executing X_vec = -(alpha/(1-alpha)*Gamma1 + Gamma2) \ gamma2;
Gamma1 = Gamma1 + Gamma2;
X_vec = -Gamma1 \ gamma2;
X = reshape(X_vec, n, N);

X_lags = extract_lags(X, 0:max_lag, max_lag);
end

function [max_abs_eig_A, A_T] = extract_sprad(W, Phi, max_lag)              % Function to extract spectral radius and first block-row of the adjacency matrix of the system if put in standard (first-order) state space form.
n = size(W, 1);
n_phi = size(Phi, 2);
n_x = max_lag - 1;
I = eye(n);
F1 = arrayfun(@(p)diag(Phi(:, p)), 1:n_phi, 'UniformOutput', 0);
A_T = [I+W+F1{1}, cell2mat(F1(2:end))-cell2mat(cellfun(@(F)F*(I+W), F1(1:end-1), 'UniformOutput', 0)), ...
    -F1{end}*(I+W), zeros(n, n*(n_x-n_phi))];
A = [A_T; eye(n*n_x), zeros(n*n_x, n)];                                     % The full adjacency matrix of the system.
max_abs_eig_A = max(abs(eig(A)));
end

function Delta = extract_Delta(H, Psi)                                      % Delta is a filter computed from H and Psi and appears naturally in the output equation when multiplying both sides by inverse noise filter.
n = size(H, 1);
n_h = size(H, 2);
n_psi = size(Psi, 2);
Psi_H = zeros(n, n_h+n_psi);                                                % The convolution (coefficients of polynomial product) of Psi and H
for p = 2:n_h+n_psi
    s_min = max(1, p-n_psi);
    s_max = min(n_h, p-1);
    Psi_H(:, p) = sum(fliplr(Psi(:, p-s_max:p-s_min)) .* H(:, s_min:s_max), 2);
end
Delta = [H, zeros(n, n_psi)] - Psi_H;
end