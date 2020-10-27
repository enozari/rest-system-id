function macro_lin_iterator_inner(section, p, n_rep)
%MACRO_LIN_ITERATOR_INNER Inner code for macro_lin_iterator, if the latter
% is run locally (not on a cluster). Otherwise, this function is not used
% as it is the same as the code in the auxiliary function of
% macro_lin_iterator.m
% 
%   Input arguments
% 
%   section: The section of macro_lin_iterator from which this function is
%   called. It determines what macro_lin should sweep over.
% 
%   p: the value of input argument p for macro_lin.
% 
%   n_rep: the number of times to repeat macro_lin.
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

switch section
    case 1
        sweep = 'fpass';
    case 3
        sweep = 'nave';
    case 5
        sweep = 'SNR';
    case 7
        sweep = 'dim';
end

% Iteratign macro_lin
R2_lin_rec = cell(n_rep, 1);
R2_nonlin_rec = cell(n_rep, 1);
h_opt_rec = cell(n_rep, 1);
for i_rep = 1:n_rep
	[R2_lin_rec{i_rep}, R2_nonlin_rec{i_rep}, h_opt_rec{i_rep}] = macro_lin(sweep, p, 1, 0);
end

% Book-keeping
R2_lin_rec = cell2mat(R2_lin_rec);
R2_nonlin_rec = cell2mat(R2_nonlin_rec);
h_opt_rec = cell2mat(h_opt_rec);
save(['macro_lin_data/section' num2str(section) '_p' num2str(p) '.mat'], 'R2_lin_rec', 'R2_nonlin_rec', ...
    'h_opt_rec');
