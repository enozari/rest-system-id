function [pval, stat, sig_thr] = my_whitetest_multivar(E, nlags, n_rand, alpha)
%MY_WHITETEST_MULTIVAR randomization-based portmanteu test of multivariate
%whiteness.
% 
%   Input arguments:
% 
%       E: multivariate residual time series (channels x observations)
% 
%       nlags: number of lags to compute auto and cross-correlations
% 
%       n_rand: number of bootstrap randomizations
% 
%       alpha: significance threshold on p-value
% 
%   Output arguments:
% 
%       pval: randomization-based p-value of the test of whiteness (null
%       hypothesis is being white)
% 
%       stat: test statistic Q
%       
%       sig_thr: randomization-based significance threshold on the test
%       statistic.
% 
%   Copyright (C) 2021, Erfan Nozari
%   All rights reserved.

if nargin < 2 || isempty(nlags)
    nlags = 20;
end
if nargin < 3 || isempty(n_rand)
    n_rand = 100;
end
if nargin < 4 || isempty(alpha)
    alpha = 0.05;
end

stat = my_whitetest_multivar_stat(E, nlags);

rng(0)
N_test = size(E, 2);
rand_stat_rec = nan(n_rand, 1);
for i_rand = 1:n_rand
    rand_stat_rec(i_rand) = my_whitetest_multivar_stat(E(:, randperm(N_test)), nlags);
end
pval = mean(stat < rand_stat_rec);
sig_thr = prctile(rand_stat_rec, (1 - alpha) * 100);
end


function stat = my_whitetest_multivar_stat(E, nlags)
[n, N_test] = size(E);
E = E - mean(E, 2);
racvf = my_xcorr(E, nlags);
reindex = reshape(n*(0:nlags-1)' + (1:n), 1, []);
warning('off', 'all')
Rhat_iT_Rhat_0_inv = racvf(:, n+1:end)' * pinv(racvf(:, 1:n));
Rhat_i_Rhat_0_invT = (reshape(racvf(:, n+reindex), n*nlags, n) * pinv(racvf(:, 1:n)))';
warning('on', 'all')
stat = (N_test - nlags) * sum(sum(Rhat_iT_Rhat_0_inv .* reshape(Rhat_i_Rhat_0_invT(:, reindex), n*nlags, n)));
end
