function out = matlab_colors(order)
%MATLAB_COLROS the default seven colors in matlab plots, as an RGB vector.
% order specifies the order in which matlab uses the colors by default.

switch order
    case 1
        out = [0 0.4470 0.7410];
    case 2
        out = [0.8500 0.3250 0.0980];
    case 3
        out = [0.9290 0.6940 0.1250];
    case 4
        out = [0.4940 0.1840 0.5560];
    case 5
        out = [0.4660 0.6740 0.1880];
    case 6
        out = [0.3010 0.7450 0.9330];
    case 7
        out = [0.6350 0.0780 0.1840];
    otherwise
        out = [0 0 0];
        warning('Order not supported, returning black ...')
end
    