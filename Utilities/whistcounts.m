function [N, cpts, h] = whistcounts(X, edges, weights, normalization, low_memory)
%WHISTCOUNTS A weighted version of MATLAB's built-in histcounts function,
% albeit with different options.
% 
%   N = whistcounts(X, edges) returns an unweighted histogram of X. edges
%   is a vector of equally spaced bin edges determining the edges of the
%   bins to which X should be discretized. X can have an arbitrary
%   dimension and size, and whistcounts operates along its first dimension.
%   Therefore, N has the same dimension and size as X except that its
%   number of rows is equal to numel(edges)-1.
% 
%   N = whistcounts(X, edges, weights) specifies an array of weights
%   corresponding to each element of X. weights must in principle be the
%   same size as X, but both of them can have missing dimensions as well,
%   provided that their binary operators (such as X .* weights) is well
%   defined in MATLAB. Each element of X is counted as much as its weight
%   in computing the histogram frequencies in N.
% 
%   N = whistcounts(X, edges, weights, normalization) also determines the
%   normalization scheme, either of 'wcont' (the raw weighted counts, and
%   the default) or 'pdf' (probability density function, so that the
%   integral of each column of N over edges will be 1).
% 
%   N = whistcounts(X, edges, weights, normalization, low_memory) further
%   determines if memory usage should be minimized. If true, a binary
%   matrix operator is broken into numel(edges)-1 iterations of a for loop,
%   increasing run time but reducing memory usage by the same factor.
% 
%   [N, cpts, h] = whistcounts(...) also returns the center points of the
%   bins cpts and their width h.
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

edges = edges(:);
if nargin < 3 || isempty(weights)
    weights = ones(size(X));
end
if nargin < 4
    normalization = 'wcount';
end
if nargin < 5
    low_memory = 1;
end

assert(max(abs(diff(edges, 2))) < 1e-10, 'The elements of edges must be equally spaced.')
h = edges(2) - edges(1);

ind = discretize(X, edges);                                                 % Discretizing X according to the edges
nbin = numel(edges) - 1;
ndims_X_and_weights = max(find(size(X) > 1, 1, 'last'), find(size(weights) > 1, 1, 'last')); % Finding the "shapred" number of dimensions of X and weights (i.e., the number of dimensions of  X .* weights)

if low_memory
    size_N = nan(1, ndims_X_and_weights);                                   % Size of N
    size_N(1) = nbin;
    for i = 2:ndims_X_and_weights
        size_N(i) = max(size(X, i), size(weights, i));
    end
    N = nan(size_N);
    colons = repmat({':'}, 1, ndims_X_and_weights-1);
    for ibin = 1:nbin
        N(ibin, colons{:}) = sum((ind == ibin) .* weights, 1);              % Weighted counts
    end
else
    dimorder = [ndims_X_and_weights+1, 2:ndims_X_and_weights, 1];           
    N = permute(sum((ind == permute((1:nbin)', dimorder)) .* weights, 1), dimorder); % Performing an automatic weighted count using matrix operations
end

switch normalization
    case 'wcount'
    case 'pdf'
        N = N ./ (sum(N, 1) * h);                                           % Normalizing the columns of N to their integrals
    otherwise
        error('Invalid normalization type. Only ''wcount'' and ''pdf'' are valid values.')
end

cpts = (edges(1:end-1) + edges(2:end)) / 2;
end