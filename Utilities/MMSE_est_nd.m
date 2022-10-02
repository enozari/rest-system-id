function Y_test = MMSE_est_nd(X_train, Y_train, X_test, N_pdf, rel_sigma, memory) 
%MMSE_EST_ND N-D minimum mean squared error (MMSE) estimator.
%
%   Y_test = MMSE_est_nd(X_train, Y_train, X_test) returns an N_test x 1 x
%   n array of MMSE estimates of the data in X_test based on the pair
%   (X_train, Y_train). X_train and Y_train must be N_train x d x n and
%   N_train x 1 x n arrays where N_train is the number of observations, n
%   is the number of channels, and d is the number of dimensions on which
%   MMSE is conditioning. X_test is N_test x d x n, where N_test is the
%   number of test observations.
% 
%   Y_test = MMSE_est_nd(X_train, Y_train, X_test, N_pdf) also provides the
%   number of discretization points for constructing probability density
%   functions from data. The range of output data in Y_train is broken into
%   N_pdf points over which the pdf of Y is estimated. 
% 
%   Y_test = MMSE_est_nd(X_train, Y_train, X_test, N_pdf, rel_sigma) also
%   determines the width of the Gaussian window used for weighting the
%   training samples around each test sample, relative to the range of the
%   elements in X_train.
% 
%   Y_test = MMSE_est_nd(X_train, Y_train, X_test, N_pdf, rel_sigma,
%   memory) also determines how much memory is available for MATLAB.
%   MMSE_est is very computationally intensive, and therefore loops are
%   avoided as much as possible and replaced with matrix operations.
%   However, this amounts to creation of large matrices that may not fit
%   into memory. The recommended value for the 'memory' input argument is
%   always minus the available memory in GB. This allows the code to
%   flexibly choose how much loops and how much matrix operations should be
%   used. memory = 0 forces the code to use no loops and assumes sufficient
%   memory is available. memory = 1 adds one loop but still assumes
%   sufficient memory is available. In either of these cases, MATLAB will
%   throw an error if it is asked to build arrays which do not fit into its
%   memory. memory = 2 adds another loop and requires the least memory, but
%   may not be efficient if more memory is available. if MATLAB still runs
%   out of memory with memory = 2, there is still room for improvement by
%   adding an additional loop over i_test = 1:N_test, but that is not
%   implemented in this version as the minimum memory requirement should be
%   sufficiently low for most devices.
% 
%   Copyright (C) 2021, Erfan Nozari
%   All rights reserved.

if nargin < 4 || isempty(N_pdf)
    N_pdf = 200;
end
if nargin < 5 || isempty(rel_sigma)
    rel_sigma = 0.01;
end
if nargin < 6 || isempty(memory)
    memory = 2;                                                             % 0 if arrays of size size(Y_test)xN_pdf^2 fit into memory, 1 if only arrays of size size(Y_test)xN_pdf fit into memory, 2 if only arrays of size size(Y_test) fit into memory. A negative number gives the amount of memory in GB so that the optimal setting is chosen by the program.
end

Y_min = min(Y_train(:));
Y_max = max(Y_train(:));
Y_pdf_edges = Y_min + (Y_max - Y_min) .* linspace(0, 1, N_pdf+1)';          % The conditional pdf of y is estimated over an N_pdf-point discretization of the the range [Ymin, Y_max].

[N_train, d, n] = size(X_train);
N_test = size(X_test, 1);

sigma = rel_sigma * mean(max(max(X_train), [], 3) - min(min(X_train), [], 3));

four_dim_size_needed = N_train * N_test * d * n * 8;
three_dim_size_needed = N_train * N_test * d * 8;
if memory == 0 || -memory*1e9 >= 1.5*four_dim_size_needed
    Y_test_pdf_weight = permute(exp(-sqrt(sum((permute(X_train, [1 4 2 3]) - permute(X_test, [4 1 2 3])).^2, 3)) ... /
        / 2/sigma^2), [1 2 4 3]);                                               % The Gaussian weights of each training data point relative to every testing data point. The result is an N_train x N_test x n array.
elseif memory > 0
    Y_test_pdf_weight = nan(N_train, N_test, n);
    parfor i = 1:n
        Y_test_pdf_weight(:, :, i) = exp(-sqrt(sum((permute(X_train(:, :, i), [1 3 2]) ... -
            - permute(X_test(:, :, i), [3 1 2])).^2, 3)) / 2/sigma^2);
    end
else
    if -memory*1e9 < 1.5*three_dim_size_needed
        warning(['Not enough memory is available. At least ' num2str(1.5*three_dim_size_needed) 'bytes of memory is required.'])
    end
    Y_test_pdf_weight = nan(N_train, N_test, n);
    n_slice = floor(-memory*1e9 / three_dim_size_needed);                     % Number of test points that can be handled in 1 slice without running out of memory.
    if n_slice > 1
        n_slice = floor(n_slice / 2);                                       % This is to ensure that we do not fill up the memory and force MATLAB to use swap memory.
    end
    n_batch = ceil(n / n_slice);                                            % Number of batches needed to handle all the N_test test points, n_slice points at a time.
    for i_batch = 1:n_batch
        i = (i_batch-1)*n_slice+1:min(n, i_batch*n_slice);
        Y_test_pdf_weight(:, :, i) = permute(exp(-sqrt(sum((permute(X_train(:, :, i), [1 4 2 3]) - permute(X_test(:, :, i), [4 1 2 3])).^2, 3)) ... /
            / 2/sigma^2), [1 2 4 3]);
    end
end

four_dim_size_needed = N_train * N_test * n * N_pdf * 8;                    % The maximum amount of memory needed if only matrix operations are used and no for loops. All sizes are in bytes
three_dim_size_needed = max(N_train, N_pdf) * N_test * n * 8;               % The amount of memory needed if one for loop is used.
two_dim_size_needed = max(N_train, N_pdf) * N_test * 8;                     % The minimum amount of memory needed if two for loops are used.
if memory == 0 || -memory*1e9 >= four_dim_size_needed
    [Y_test_pdf, Y_test_pdf_cpts, Y_test_pdf_binwidth] = whistcounts(Y_train, Y_pdf_edges, Y_test_pdf_weight, ...
        'pdf', 0);
    Y_test = permute(sum(Y_test_pdf_cpts .* Y_test_pdf, 1) * Y_test_pdf_binwidth, [2 1 3]);
elseif memory == 1 || -memory*1e9 >= three_dim_size_needed
    [Y_test_pdf, Y_test_pdf_cpts, Y_test_pdf_binwidth] = whistcounts(Y_train, Y_pdf_edges, Y_test_pdf_weight, ...
        'pdf', 1);
    Y_test = permute(sum(Y_test_pdf_cpts .* Y_test_pdf, 1) * Y_test_pdf_binwidth, [2 1 3]);
elseif memory == 2
    Y_test = nan(N_test, 1, n);
    parfor i = 1:n
        [Y_test_pdf, Y_test_pdf_cpts, Y_test_pdf_binwidth] = whistcounts(Y_train(:, :, i), Y_pdf_edges, ...
            Y_test_pdf_weight(:, :, i), 'pdf', 1);
        Y_test(:, :, i) = sum(Y_test_pdf_cpts .* Y_test_pdf, 1)' * Y_test_pdf_binwidth;
    end
else
    if -memory*1e9 < two_dim_size_needed
        warning(['Out of memory. At least ' num2str(two_dim_size_needed) ' bytes of memory is needed.'])
    end
    Y_test = nan(N_test, 1, n);
    n_slice = floor(-memory*1e9 / two_dim_size_needed);                     % Number of test points that can be handled in 1 slice without running out of memory.
    if n_slice > 1
        n_slice = floor(n_slice / 2);                                       % This is to ensure that we do not fill up the memory and force MATLAB to use swap memory.
    end
    n_batch = ceil(n / n_slice);                                            % Number of batches needed to handle all the N_test test points, n_slice points at a time.
    for i_batch = 1:n_batch
        i = (i_batch-1)*n_slice+1:min(n, i_batch*n_slice);
        [Y_test_pdf, Y_test_pdf_cpts, Y_test_pdf_binwidth] = whistcounts(...
            Y_train(:, :, i), Y_pdf_edges, Y_test_pdf_weight(:, :, i), 'pdf', 1);
        Y_test(:, :, i) = permute(sum(Y_test_pdf_cpts .* Y_test_pdf, 1) * Y_test_pdf_binwidth, [2 1 3]);
    end
end

for i = 1:n
    Y_test(isnan(Y_test(:, 1, i)), 1, i) = mean(Y_test(:, 1, i));           % This replaces the points that were unpredictable due to being far from all training points with the average of their channel, which is the unconditional MMSE predictor.
end