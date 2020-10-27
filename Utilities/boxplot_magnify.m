function boxplot_magnify(hb, scale)
%BOXPLOT_MAGNIFY Function to magnify the size of compact boxplot boxes.
% 
%   boxplot_magnify(hb, scale) magnifies the boxplot in the handle hb by
%   the given scale.

for hbi = reshape(hb, 1, [])
    if ~isequal(hbi.Tag, 'boxplot')
        continue
    end
    tags = get(hbi.Children, 'Tag');
    for hc = hbi.Children(strcmp(tags, 'Box'))'
        hc.LineWidth = hc.LineWidth * scale;
        box_linewidth = hc.LineWidth;
        box_color = hc.Color;
    end
    for hc = hbi.Children(strcmp(tags, 'MedianOuter'))'
        hc.MarkerSize = box_linewidth * 1.2;
    end
    for hc = hbi.Children(strcmp(tags, 'MedianInner'))'
        hc.MarkerSize = box_linewidth * 1.2;
        hc.MarkerEdgeColor = box_color;
    end
    for hc = hbi.Children(strcmp(tags, 'Whisker'))'
        hc.LineWidth = box_linewidth / 4;
    end
    for hc = hbi.Children(strcmp(tags, 'Outliers'))'
        hc.MarkerSize = box_linewidth * 0.4;
    end
end