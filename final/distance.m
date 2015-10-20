function [y] = distance(r,l);

y = sqrt(sum((r-l) .^ 2));