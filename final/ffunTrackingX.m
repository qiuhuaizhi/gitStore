function [y] = ffunTrackingX(x,S,t);
% PURPOSE : Process model function.
% INPUTS  : - x:  The evaluation point in the domain.
% OUTPUTS : - y: The value of the function at x.
% AUTHORS  : 
% DATE     : 


if nargin < 2, error('Not enough input arguments.'); end

% y = x.^2/20; 
y(1,1) = x(1,1)-S(1,1);










