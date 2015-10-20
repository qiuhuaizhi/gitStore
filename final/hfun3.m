function [y] = hfun2(x,t,Q);
% PURPOSE : Measurement model function.
% INPUTS  : - x:  The evaluation point in the domain.
% OUTPUTS : - y: The value of the function at x.
% AUTHORS  : 
% DATE     : 

if nargin < 2, error('Not enough input arguments.'); end

w = sqrt(Q)*randn(size(x));
y = 0.5.*x +  w + 10;








