function c = my_xcorr(x, nlags)
%MY_XCORR Pairwise cross-correlation. Similar to MATLAB's xcorr, albeit
%with less options, but more computational efficiency.
% 
%   x: input signal (channels x observations)
% 
%   nlags: number of time lags to use for the calculation of auto and
%   cross-correlations
% 
%   c: matrix of auto and cross-correlations.

x_lags = cell2mat(arrayfun(@(lag)x(:, 1+lag:end-nlags+lag), (0:nlags)', 'UniformOutput', 0));
x_0 = x(:, 1:end-nlags);
N = size(x, 2);
c = x_0 * x_lags' / (N - nlags);
