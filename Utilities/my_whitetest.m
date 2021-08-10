function [pval, stat] = my_whitetest(E, nlags)
%MY_WHITETEST chi-squre test of (univariate) whiteness.
% 
%   [pval, stat] = my_whitetest(E) returns the p-value for the chi-squared
%   test of whiteness of each row of E. stat is the corresponding test
%   statistic. Both pval and stat are column vectors having as many
%   elements as the number of rows of E.
% 
%   [pval, stat] = my_whitetest(E, nlags) additionally determines the
%   number of lags in the autocorrelation. The default is 20.
% 
%   Copyright (C) 2021, Erfan Nozari
%   All rights reserved.

if nargin < 2
    nlags = 20;
end

if isvector(E)
    E = E(:)';
end

[n, N] = size(E);
pval = nan(1, n);
stat = nan(1, n);
for i = 1:n
    acf = my_autocorr(E(i, :)', nlags);
    stat(i) = N * sum(acf(2:end).^2);
    pval(i) = my_chi2cdf(stat(i), nlags, 'upper');
end
end

%% Auxiliary functions
function p = my_chi2cdf(x, v, type)                                         % In place of MATLAB's built-in chi2cdf function, hand coded to avoid using additional toolboxes.
p = gammainc(x/2, v/2, type);
end