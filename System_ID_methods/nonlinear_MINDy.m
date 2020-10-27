function [model, R2, whiteness_p, Y_hat] = nonlinear_MINDy(Y, TR, doPreProc, lambda, use_parallel, ...
    test_range)
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
if nargin < 5 || isempty(use_parallel)
    use_parallel = 1;
end
if nargin <6 || isempty(test_range)
    test_range = [0.8 1];
end

%% Organizing data into separate train and test segments
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);

%% Model fitting
addpath(genpath('MINDy-Beta-master'));
[~, MINDy_Out] = evalc('MINDy_Simple_regparam(Y_train_cell, [], TR, doPreProc, lambda{:})');    % Use of evalc prevents output written to workspace from MINDy_Simple.

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

%% Cross-validated one step ahead prediction
n_test = numel(Y_test_cell);
Y_test_hat_cell = arrayfun(@(N_test)nan(n, N_test), N_test_vec, 'UniformOutput', 0);
if isequal(doPreProc, 'y')                                                  % In this case since Wiener deconvlution with the nominal HRF is applied, one step ahead prediction should first be done at the level of internal states (which in turn need to be estimated first using HRF_deconv) and then mapped to output Y using the forward convolution with the same nominal HRF.
    for i_test = 1:n_test                                                   % Iterating over each testing segment separately since testing segments are by assumption not from continuous recordings
        Y_test = Y_test_cell{i_test};
        N_test = N_test_vec(i_test);
        Y_test_hat = nan(n, N_test);
        if use_parallel
            parfor t = 1:N_test-1
                X_test_hat_1tot = HRF_deconv([zeros(n, 2-t) Y_test(:, 1:t)], TR); % Wiener deconvolution, as used in MINDy
                MINDyInt_Out = MINDyInt(MINDy_Out, X_test_hat_1tot(:, end-1), 1, TR, 0, TR, 0); % One step ahead prediction of state
                X_test_hat_1tot(:, end) = MINDyInt_Out(:, 2);               % The estimate of the last column of X_test_hat_1tot, corresponding to time t, based on deconvolution should be discarded (because HRF_impulse_resp has a delay of 1, i.e., is nonzero from time 1 not from time 0) and estimated from the previous column using one step ahead prediction
                Y_test_hat(:, t+1) = sum(fliplr(HRF_impulse_resp{i_test}(1:size(X_test_hat_1tot, 2))) .* X_test_hat_1tot, 2); % Convolution with the same nominal HRF used in MINDy
            end
        else
            for t = 1:N_test-1
                X_test_hat_1tot = HRF_deconv([zeros(n, 2-t) Y_test(:, 1:t)], TR);
                MINDyInt_Out = MINDyInt(MINDy_Out, X_test_hat_1tot(:, end-1), 1, TR, 0, TR, 0);
                X_test_hat_1tot(:, end) = MINDyInt_Out(:, 2);
                Y_test_hat(:, t+1) = sum(fliplr(HRF_impulse_resp{i_test}(1:size(X_test_hat_1tot, 2))) .* X_test_hat_1tot, 2);
            end
        end
        Y_test_hat_cell{i_test} = Y_test_hat;
    end
else                                                                        % In this case the model runs directly on the output (the state and output are the same) and therefore one step ahead prediction can be directly obtained by running the neural mass model forwared.
    for i_test = 1:n_test
        Y_test = Y_test_cell{i_test};
        MINDyInt_Out = MINDyInt(MINDy_Out, Y_test(:, 1:end-1), 1, TR, 0, TR, 0);
        Y_test_hat = [nan(n, 1), squeeze(MINDyInt_Out(:, 2, :))];
        Y_test_hat_cell{i_test} = Y_test_hat;
    end
end

Y_test_full = cell2mat(Y_test_cell);
Y_test_hat_full = cell2mat(Y_test_hat_cell);
nan_ind = any(isnan(Y_test_hat_full));                                      % Index of time points at which a cross-validated prediction is not available, either because they correspond to training samples, or they are the first sample of their recording
E_test = Y_test_hat_full(:, ~nan_ind) - Y_test_full(:, ~nan_ind);           % (Output) prediction error
R2 = 1 - sum(E_test.^2, 2) ./ sum((Y_test_full(:, ~nan_ind) - mean(Y_test_full(:, ~nan_ind), 2)).^2, 2);
whiteness_p = my_whitetest(E_test');

Y_hat = [nan(n, test_ind(1)), Y_test_hat_full, nan(n, N-test_ind(end))];    % Appending Y_test_hat with NaNs before and after corresponding to training time points.
if iscell(Y)
    Y_hat = mat2cell(Y_hat, n, diff(break_ind));
end