function [model, R2, whiteness, Y_hat, runtime] = nonlinear_MINDy(Y, TR, ...
    doPreProc, lambda, k, use_parallel, test_range)
%NONLINEAR_MINDY Fitting and cross-validating a nonlinear neural mass model
% using the MINDy algorithm presented in:
% Singh M, Braver T, Cole M, Ching S. Individualized Dynamic Brain Models:
% Estimation and Validation with Resting-State fMRI. bioRxiv.
%
%   Input arguments
% 
%   Y: a data matrix or cell array of data matrices used for system
%   identification. Each element of Y (or Y itself) is one scan, with
%   channels along the first dimension and time along the second dimension.
%   This is the only mandatory input.
% 
%   TR: sampling time.
% 
%   doPreProc: binary flag ('y' or 'n') to determine whether MINDy's
%   preprocessing (Wiener deconvolution and z-scoring) should be done.
% 
%   lambda: hyperparameters lambda1-lambda4 of the MINDy algorithm. Leave
%   empty to use the defaults.
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
    TR = 0.002;
end
if nargin < 3 || isempty(doPreProc)
    doPreProc = 'n';
end
if nargin < 4 || isempty(lambda)
    lambda = {};
end
assert(iscell(lambda) && (numel(lambda) == 4 || isempty(lambda)), ...
    'lambda should be a cell array with no or 4 scalar elements.')
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

%% Model fitting
runtime_train_start = tic;

MINDy_Out = MINDy_Simple_regparam(Y_train_cell, TR, [], doPreProc, lambda{:});

if isequal(doPreProc, 'y')
    model.eq = ['$$x(t) - x(t-1) = (W \psi_\alpha(x(t-1)) - D x(t-1)) \Delta T + e_1(t) \\ ' ...
        'y(t) = H(q) x(t) + e_2(t) \\ ' ...
        '\psi_\alpha(x) = [\sqrt{alpha_i^2 + (b x_i + 0.5)^2} - \sqrt{alpha_i^2 + (b x_i - 0.5)^2}]_{i=1}^n \\ ' ...
        'H(q) = \sum_{p\ge1} h_p q^{-p}$$'];
    model.W = MINDy_Out.Param{5};
    model.alpha = MINDy_Out.Param{2};
    model.b = 20/3;
    model.D = MINDy_Out.Param{6};
    model.DeltaT = TR;
    HRF_impulse_resp = arrayfun(@(N_test)nominal_HRF_coeffs(TR, N_test)', N_test_vec, 'UniformOutput', 0);
    [~, max_ind] = max(cellfun(@numel, HRF_impulse_resp));
    model.h = HRF_impulse_resp{max_ind};
else
    model.eq = ['$$y(t) - y(t-1) = (W \psi_\alpha(y(t-1)) - D y(t-1)) \Delta T + e(t) \\ ' ...
        '\psi_\alpha(y) = [\sqrt{alpha_i^2 + (b y_i + 0.5)^2} - \sqrt{alpha_i^2 + (b y_i - 0.5)^2}]_{i=1}^n$$'];
    model.W = MINDy_Out.Param{5};
    model.alpha = MINDy_Out.Param{2};
    model.b = 20/3;
    model.D = MINDy_Out.Param{6};
    model.DeltaT = TR;
end

runtime.train = toc(runtime_train_start);

%% Cross-validated k-step ahead prediction
runtime_test_start = tic;

n_test = numel(Y_test_cell);
Y_test_hat_cell = arrayfun(@(N_test)nan(n, N_test), N_test_vec, 'UniformOutput', 0);
if isequal(doPreProc, 'y')                                                  % In this case since Wiener deconvlution with the nominal HRF is applied, one step ahead prediction should first be done at the level of internal states (which in turn need to be estimated first using HRF_deconv) and then mapped to output Y using the forward convolution with the same nominal HRF.
    H1 = 6;
    H2 = 1;
    h_HRF = MINDy_MakeHRF_H1H2(H1, H2);
    HRF_deconv_SNR = 0.02;
    for i_test = 1:n_test                                                   % Iterating over each testing segment separately since testing segments are by assumption not from continuous recordings
        Y_test = Y_test_cell{i_test};
        N_test = N_test_vec(i_test);
        Y_test_hat = nan(n, N_test);
        if use_parallel
            parfor t = k+1:N_test
                H_HRF = fft(h_HRF(TR * (0:t-k-1)), [], 2);
                X_test_hat_1totmk = real(ifft(fft(Y_test(:, 1:t-k), [], 2) ... .*
                    .* (conj(H_HRF) ./ (HRF_deconv_SNR + abs(H_HRF).^2)), [] ,2));
                X_test_hat_1pitotmkpi = X_test_hat_1totmk;                  % Initializing at i = 0
                for i = 1:k
                    X_test_hat_1pitotmkpi = (1 - MINDy_Out.Param{6}) .* X_test_hat_1pitotmkpi ... +
                        + MINDy_Out.Param{5} * MINDy_Out.Tran(X_test_hat_1pitotmkpi);
                end
                X_test_hat_1pktot = X_test_hat_1pitotmkpi ... - 
                    - (1 - MINDy_Out.Param{6}).^k .* X_test_hat_1totmk;     % Removing the linear part to add later after applying the HRF
                Y1_test_hat_1pktot = real(ifft(fft(X_test_hat_1pktot, [], 2) .* H_HRF, [], 2));
                Y_test_hat(:, t) = Y1_test_hat_1pktot(:, end) ... +
                    + (1 - MINDy_Out.Param{6}).^k .* Y_test(:, t-k);  
            end
        else
            for t = k+1:N_test
                H_HRF = fft(h_HRF(TR * (0:t-k-1)), [], 2);
                X_test_hat_1totmk = real(ifft(fft(Y_test(:, 1:t-k), [], 2) ... .*
                    .* (conj(H_HRF) ./ (HRF_deconv_SNR + abs(H_HRF).^2)), [] ,2));
                X_test_hat_1pitotmkpi = X_test_hat_1totmk;                  % Initializing at i = 0
                for i = 1:k
                    X_test_hat_1pitotmkpi = (1 - MINDy_Out.Param{6}) .* X_test_hat_1pitotmkpi ... +
                        + MINDy_Out.Param{5} * MINDy_Out.Tran(X_test_hat_1pitotmkpi);
                end
                X_test_hat_1pktot = X_test_hat_1pitotmkpi ... - 
                    - (1 - MINDy_Out.Param{6}).^k .* X_test_hat_1totmk;     % Removing the linear part to add later after applying the HRF
                Y1_test_hat_1pktot = real(ifft(fft(X_test_hat_1pktot, [], 2) .* H_HRF, [], 2));
                Y_test_hat(:, t) = Y1_test_hat_1pktot(:, end) ... +
                    + (1 - MINDy_Out.Param{6}).^k .* Y_test(:, t-k);  
            end
        end
        Y_test_hat_cell{i_test} = Y_test_hat;
    end
else                                                                        % In this case the model runs directly on the output (the state and output are the same) and therefore one step ahead prediction can be directly obtained by running the neural mass model forwared.
    for i_test = 1:n_test
        Y_test = Y_test_cell{i_test};
        Y_test_plus_hat = Y_test(:, 1:end-1) + MINDy_Out.FastFun(Y_test(:, 1:end-1));
        for i = 2:k
            Y_test_plus_hat = Y_test_plus_hat(:, 1:end-1) ... +
                + MINDy_Out.FastFun(Y_test_plus_hat(:, 1:end-1));
        end
        Y_test_hat = [nan(n, k), Y_test_plus_hat];
        Y_test_hat_cell{i_test} = Y_test_hat;
    end
end

Y_test_full = cell2mat(Y_test_cell);
Y_test_hat_full = cell2mat(Y_test_hat_cell);
Y_hat = [nan(n, test_ind(1)), Y_test_hat_full, nan(n, N-test_ind(end))];    % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind));
end

runtime.test = toc(runtime_test_start);
runtime.total = runtime.train + runtime.test;

nan_ind = any(isnan(Y_test_hat_full));                                      % Index of time points at which a cross-validated prediction is not available, either because they correspond to training samples, or they are the first sample of their recording
E_test = Y_test_hat_full(:, ~nan_ind) - Y_test_full(:, ~nan_ind);           % (Output) prediction error
R2 = 1 - sum(E_test.^2, 2) ./ sum((Y_test_full(:, ~nan_ind) - mean(Y_test_full(:, ~nan_ind), 2)).^2, 2);
[whiteness.p, whiteness.stat, whiteness.sig_thr] = my_whitetest_multivar(E_test);

