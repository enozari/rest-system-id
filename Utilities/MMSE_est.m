function Y_test = MMSE_est(X_train, Y_train, X_test, N_pdf, pdf_weight, memory) 
%MMSE_EST pairwise minimum mean squared error (MMSE) estimator.
%
%   Y_test = MMSE_est(X_train, Y_train, X_test) gives a 3D array of
%   pairwise MMSE estimates of the data in X_test based on the pair
%   (X_train, Y_train). X_train and Y_train must be matrices (including
%   column vectors) with dimension N_train x n where N_train is the number
%   of observations and n is the number of channels. X_test is N_test x n,
%   where N_test is the number of test observations and need not be the
%   same as N_train. Y_test is N_test x n x n whose (i, j, k)th element is
%   the MMSE estimate of the output (y) of the k'th channel given X_test(i,
%   j) based on the training data in X_train(:, j) and Y_train(:, k). In
%   other words, the slice Y_test(i, :, :) contains all the pairwise
%   predictions of the output channels y at "time" i from the i'th
%   observation of each of the input channels at time i in X_test(i, :).
% 
%   Y_test = MMSE_est(X_train, Y_train, X_test, N_pdf) also provides the
%   number of discretization points for constructing probability density
%   functions from data. The range of output data in Y_train is broken into
%   N_pdf points over which the pdf of Y is estimated. The default is N_pdf
%   = 200.
% 
%   Y_test = MMSE_est(X_train, Y_train, X_test, N_pdf, pdf_weight) also
%   determines the type and parameter used for density estimation.
%   pdf_weight is a struct with a mandatory field method that can be either
%   'normpdf' for estimation using Gaussian windows or 'knn' for estimation
%   using k-nearest neighbors method. if pdf_weight.method = 'normpdf',
%   then pdf_weight must have a second field rel_sigma which determines the
%   width of the Gaussian window relative to the range of x. if
%   pdf_weight.method = 'knn', then pdf_weight must have a second field
%   rel_K which determines the number of nearest neighbors relative to the
%   total number of observations.
% 
%   Y_test = MMSE_est(X_train, Y_train, X_test, N_pdf, pdf_weight, memory)
%   also determines how much memory is available for MATLAB. MMSE_est is
%   very computationally intense, and therefore loops are avoided as much
%   as possible (and replaced with matrix operations). However, this
%   amounts to creation of large matrices that may not fit into memory.
%   The recommended value is always minus the available memory in GB. This
%   allows the code to flexibly choose how much for loops and how much
%   matrix operations should be used. memory = 0 forces the code to use no
%   for loops and assumes sufficient memory is available. memory = 1 adds
%   one layer of for loop but still assumes sufficient memory is available.
%   In either of these latter two cases, MATLAB will throw an error if it
%   is asked to build arrays which do not fit into its memory. memory = 2
%   adds the maximum possible number of loops and requires the least memory
%   (but may not be efficient if more memory is available). Not much can be
%   done if MATLAB still runs out of memory with memory = 2, other than
%   reducing the number of observations.
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

if nargin < 4 || isempty(N_pdf)
    N_pdf = 200;
end
if nargin < 5 || isempty(pdf_weight)
    pdf_weight.method = 'normpdf';
    pdf_weight.rel_sigma = 0.01;
end
if nargin < 6 || isempty(memory)
    memory = 2;                                                             % 0 if arrays of size size(Y_test)xN_pdf^2 fit into memory, 1 if only arrays of size size(Y_test)xN_pdf fit into memory, 2 if only arrays of size size(Y_test) fit into memory. A negative number gives the amount of memory in GB so that the optimal setting is chosen by the program.
end

Y_min = min(Y_train(:));
Y_max = max(Y_train(:));
Y_pdf_edges = Y_min + (Y_max - Y_min) .* linspace(0, 1, N_pdf+1)';          % The conditional pdf of y is estimated over an N_pdf-point discretization of the the range [Ymin, Y_max].

[N_train, n] = size(X_train);
N_test = size(X_test, 1);
warning('off')
switch pdf_weight.method
    case 'normpdf'
        pdf_weight.sigma = pdf_weight.rel_sigma * (max(X_train(:)) - min(X_train(:)));
        Y_test_pdf_weight = exp(-(permute(X_train, [1 3 2]) - permute(X_test, [3 1 2])).^2 / 2/pdf_weight.sigma^2); % The Gaussian weights of each training data point relative to every testing data point. Note that the MMSE estimation is pairwise, so the distance between data points is a scalar distance (per channel), not a vector distance.
    case 'knn'
        pdf_weight.K = round(pdf_weight.rel_K * N_train);
        Y_test_pdf_weight = nan(N_train, N_test, n);
        for j = 1:n
            knn_ind = knnsearch(X_train(:, j), X_test(:, j), 'K', pdf_weight.K); % The idices of the k nearest neighbors of each X_test(i, j) from X_train(:, j).
            Y_test_pdf_weight(:, :, j) = any(permute(knn_ind, [3 1 2]) == (1:N_train)', 3); % Binary weights with a 1 corresponding to k nearest neighbors.
        end
end

five_dim_size_needed = N_train * N_test * n * n * N_pdf * 8;                % The maximum amount of memory needed if only matrix operations are used and no for loops. All sizes are in bytes
four_dim_size_needed = max(N_train, N_pdf) * N_test * n * n * 8;            % The amount of memory needed if one for loop is used.
three_dim_size_needed = max([N_train N_test N_pdf]) * n * n * 8;            % The minimum amount of memory needed if two for loops are used.
if memory == 0 || -memory*1e9 >= five_dim_size_needed
    [Y_test_pdf, Y_test_pdf_cpts, Y_test_pdf_binwidth] = whistcounts(...
        permute(Y_train, [1 4 3 2]), Y_pdf_edges, Y_test_pdf_weight, 'pdf', 0);
    Y_test = permute(sum(Y_test_pdf_cpts .* Y_test_pdf, 1) * Y_test_pdf_binwidth, [2 3 4 1]);
elseif memory == 1 || -memory*1e9 >= four_dim_size_needed
    [Y_test_pdf, Y_test_pdf_cpts, Y_test_pdf_binwidth] = whistcounts(...
        permute(Y_train, [1 4 3 2]), Y_pdf_edges, Y_test_pdf_weight, 'pdf', 1);
    Y_test = permute(sum(Y_test_pdf_cpts .* Y_test_pdf, 1) * Y_test_pdf_binwidth, [2 3 4 1]);
elseif memory == 2
    Y_test = nan(N_test, n, n);
    parfor i_test = 1:N_test
        [Y_test_pdf, Y_test_pdf_cpts, Y_test_pdf_binwidth] = whistcounts(...
            permute(Y_train, [1 4 3 2]), Y_pdf_edges, Y_test_pdf_weight(:, i_test, :), 'pdf', 1);
        Y_test(i_test, :, :) = permute(sum(Y_test_pdf_cpts .* Y_test_pdf, 1) * Y_test_pdf_binwidth, [2 3 4 1]);
    end
elseif -memory*1e9 >= three_dim_size_needed
    Y_test = nan(N_test, n, n);
    n_slice = floor(-memory*1e9 / three_dim_size_needed);                   % Number of test points that can be handled in 1 slice without running out of memory.
    if n_slice > 1
        n_slice = floor(n_slice / 2);                                       % This is to ensure that we do not fill up the memory and force MATLAB to use swap memory.
    end
    n_batch = ceil(N_test / n_slice);                                       % Number of batches needed to handle all the N_test test points, n_slice points at a time.
    for i_batch = 1:n_batch
        i_test = (i_batch-1)*n_slice+1:min(N_test, i_batch*n_slice);
        [Y_test_pdf, Y_test_pdf_cpts, Y_test_pdf_binwidth] = whistcounts(...
            permute(Y_train, [1 4 3 2]), Y_pdf_edges, Y_test_pdf_weight(:, i_test, :), 'pdf', 1);
        Y_test(i_test, :, :) = permute(sum(Y_test_pdf_cpts .* Y_test_pdf, 1) * Y_test_pdf_binwidth, [2 3 4 1]);
    end
else
    error(['Out of memory. At least ' num2str(three_dim_size_needed) ' bytes of memory is needed.'])
end

    
