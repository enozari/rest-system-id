function h_0 = nominal_HRF_coeffs(TR, n_h)
%NOMINAL_HRF_COEFFS Nominal HRF impulse response coefficients, as used in 
% Singh M, Braver T, Cole M, Ching S. Individualized Dynamic Brain Models:
% Estimation and Validation with Resting-State fMRI. bioRxiv.
% 
%   h_0 = nominal_HRF_coeffs(TR, n_h) returns the impulse response of a
%   nominal HRF with transfer function H(s) = 1/(s + 1)^6 - 1/(6*(s +
%   1)^16), sampling time TR and number of impulse response time points
%   n_h.

tspan = (1:n_h)' * TR;
h_0 = (tspan.^5/factorial(5) - tspan.^15/6/factorial(15)) .* exp(-tspan);
end