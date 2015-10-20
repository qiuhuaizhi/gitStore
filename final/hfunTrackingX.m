function [y] = hfunTrackingX(x,t);
% PURPOSE : Measurement model function.
% INPUTS  : - x:  The evaluation point in the domain.
% OUTPUTS : - y: The value of the function at x.
% AUTHORS  : 
% DATE     : 

if nargin < 2, error('Not enough input arguments.'); end

F = [1 1  ; 0 1];
y = x(1,:)*F';






