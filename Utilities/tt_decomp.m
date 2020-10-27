function [Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range)
%TT_DECOMP Decomposing one or more scans into training and test segments.
%
%   [Y_train_cell, Y_test_cell] = tt_decomp(Y, test_range) returns two cell
%   arrays that contain, respectively, continuously recorded training and
%   test segments from the input Y based on the two-element vector
%   test_range. Y can be an numeric array or cell array of numeric arrays,
%   in both cases the time should be along the second dimension of the
%   numeric array(s) (number of channels x number of observations). If Y is
%   a cell array, each element of it is assumed to be continuously
%   recorded. test_range contains two elements in the range [0, 1], the
%   second being larger than the first, determining the portion of the data
%   in Y that should be used for test. The remainder of Y is used for
%   training.
% 
%   [Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] =
%   tt_decomp(Y, test_range) additionally returns, respectively, the index
%   of the points at which the time series are broken, the index of the
%   test points, a vector containing the length of test segments in
%   Y_test_cell, number of channels, and total number of time points.
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.


switch class(Y)
    case 'double'
        break_ind = [0, size(Y, 2)];
    case 'cell'
        Y = reshape(Y, 1, []);
        break_ind = [0, cumsum(cellfun(@(Y)size(Y, 2), Y))];
        Y = cell2mat(Y);
    otherwise
        error('Y should be either a single matrix or a cell array of matrices.');
end

[n, N] = size(Y);
test_ind = round(test_range * N);
test_ind = [test_ind(1), break_ind(break_ind > test_ind(1) & break_ind < test_ind(2)), test_ind(2)];
Y_test_cell = arrayfun(@(begin_ind, end_ind)Y(:, begin_ind+1:end_ind), test_ind(1:end-1), test_ind(2:end), 'UniformOutput', 0);
N_test_vec = cellfun(@(Y)size(Y, 2), Y_test_cell);
break_ind_ext = sort(unique([break_ind test_ind]), 'ascend');
train_begin_ind = [break_ind_ext(break_ind_ext < test_ind(1)), break_ind_ext(break_ind_ext >= test_ind(end) & break_ind_ext < N)];
train_end_ind = [break_ind_ext(break_ind_ext <= test_ind(1) & break_ind_ext > 0), break_ind_ext(break_ind_ext > test_ind(end))];
Y_train_cell = arrayfun(@(begin_ind, end_ind)Y(:, begin_ind+1:end_ind), train_begin_ind, train_end_ind, 'UniformOutput', 0);