function [y] = ffun_UNGM(x,t);
% PURPOSE : Process model function.
% INPUTS  : - x:  The evaluation point in the domain.
% OUTPUTS : - y: The value of the function at x.
% AUTHORS  : 
% DATE     : 


if nargin < 2, error('Not enough input arguments.'); end

y = x.^2/20; 









