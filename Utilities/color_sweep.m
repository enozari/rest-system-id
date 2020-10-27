function colors = color_sweep(base_color, n_lighter, n_darker, max_light_ratio, max_dark_ratio)
%COLOR_SWEEP Function to generate a sweep of colors of different darkness
% from a base color.
%
%   colors = color_sweep(base_color, n_lighter, n_darker) returns an
%   (n_lighter+n_darker+1)x3 matrix of RGB color values (all entries in [0,
%   1]) consisting of the color base_color in the middle, n_lighter lighter
%   colors of the same base before it and n_darker darker colors of the
%   same base after it.
% 
%   colors = color_sweep(base_color, n_lighter, n_darker, max_light_ratio,
%   max_dark_ratio) additionally specifies how light should the lightest
%   color (max_light_ratio = 0 indicating white, max_light_ratio = 1
%   indicating the base_color) and how dark should the darkest color
%   (max_dark_ratio = 0 indicating black, max_dark_ratio = 1 indicating
%   base_color) be.

if nargin < 4 || isempty(max_light_ratio)
    max_light_ratio = 0.3;
end
if nargin < 5 || isempty(max_dark_ratio)
    max_dark_ratio = 0;
end
light_ratio = linspace(max_light_ratio, 1, n_lighter+1);
light_ratio(end) = [];
dark_ratio = fliplr(linspace(max_dark_ratio, 1, n_darker+1));
colors = [1 - (1 - base_color) .* light_ratio'; base_color .* dark_ratio'];