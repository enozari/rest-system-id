function X = HRF_deconv(Y, TR, HRF_deconv_SNR)
%HRF_deconv Function to apply standard Wiener deconvolution with a standard
% HRF impulse response based on
% Singh M, Braver T, Cole M, Ching S. Individualized Dynamic Brain Models:
% Estimation and Validation with Resting-State fMRI. bioRxiv.
%
%   X = HRF_deconv(Y) returns the deconvolved version of Y. See
%   documentation for deconvwnr for details.
% 
%   X = HRF_deconv(Y, TR) additionally specifies the sampling time TR. The
%   default is HCP TR = 0.72.
% 
%   X = HRF_deconv(Y, TR, HRF_deconv_SNR) also specifies the expected
%   signal to noise ratio. The default level is 0.02.

if nargin < 2
    TR = 0.72;
end
if nargin < 3
    HRF_deconv_SNR = 0.02;
end

a1 = 6;
a2 = 16;
b1 = 1;
b2 = 1;
c = 1/6;
HRF_impulse_resp = @(t)(t.^(a1-1).*exp(-b1*t)*(b1^a1))/gamma(a1) - c*((t.^(a2-1).*exp(-b2*t)*b2^a2)/gamma(a2)); % The canonical HRF impluse response
tspan = TR * (0:(size(Y, 2)-1));

X = deconvwnr(Y, HRF_impulse_resp(tspan), HRF_deconv_SNR);
end