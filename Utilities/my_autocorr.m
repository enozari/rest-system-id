function acf = my_autocorr(y, nlags)
%MY_AUTOCORR The same as MATLAB's built-in autocorr, albeit with less
% options, hand coded to avoid using additional toolboxes.

ybar = mean(y);
c = arrayfun(@(k)sum((y(1:end-k) - ybar) .* (y(1+k:end) - ybar)), 0:nlags);
acf = c / c(1);