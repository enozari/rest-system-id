function [p, R] = my_whitetest(x, nlags)
%MY_WHITETEST chi-squre test of whiteness.
% 
%   [p, R] = my_whitetest(x) returns the p-value for the chi-squared test
%   of whiteness of each column of x. R is the corresponding test
%   statistic. Both p and R are row vectors having as many elements as the
%   number of rows of x.
% 
%   [p, R] = my_whitetest(x, nlags) additionally determines the number of
%   lags in the autocorrelation. The default is 20.
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

if nargin < 2
    nlags = 20;
end

if isvector(x)
    x = x(:);
end

[N, n] = size(x);
p = nan(1, n);
R = nan(1, n);
for i = 1:n
    acf = my_autocorr(x(:, i), nlags);
    R(i) = N * sum(acf(2:end).^2);
    p(i) = my_chi2cdf(R(i), nlags, 'upper');
end
end

%% Auxiliary functions
function p = my_chi2cdf(x, v, type)                                         % In place of MATLAB's built-in chi2cdf function, hand coded to avoid using additional toolboxes.
p = gammainc(x/2, v/2, type);
end