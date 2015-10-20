function [y] = hfun(x,t);
% PURPOSE : Measurement model function.
% INPUTS  : - x:  The evaluation point in the domain.
% OUTPUTS : - y: The value of the function at x.
% AUTHORS  : 
% DATE     : 

if nargin < 2, error('Not enough input arguments.'); end

% if t<=30
%  y = (x.^(2))/5; 
% else
%   y = -2 + x/2; 
% end;
perfi = 20;
d0 = 1;
alpht = 2.5;

y = (perfi * d0 * alpht) / (distance(r - l)) ^ alpht + v;





