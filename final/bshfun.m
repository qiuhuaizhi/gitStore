function [y] = bshfun(x,u,t);
% PURPOSE : Measurement model function.
% INPUTS  : - x:  The evaluation point in the domain.
% OUTPUTS : - y:  The value of the function at x.

% AUTHORS  : Nando de Freitas      (jfgf@cs.berkeley.edu)
%            Rudolph van der Merwe (rvdmerwe@ece.ogi.edu)
% DATE     : 10 March 2000

if nargin < 3, error('Not enough input arguments.'); end

[dim,np] = size(x);

y=zeros(2,np);

for j=1:np,

  r   = x(1,j);   % Risk free interest rate.
  sig = x(2,j);   % Volatility.
  S   = u(1);
  tm  = u(2);
  
  d1 = ( log(S) + (r+0.5*(sig^2))*tm ) / (sig * (tm^0.5));
  d2 = d1 - sig * (tm^0.5);
  % Compute call prices:
  y(1,j) = S*normcdf(d1) - exp(-r*tm)*normcdf(d2);
  % Compute put prices:
  y(2,j) = - S*normcdf(-d1) + exp(-r*tm)*normcdf(-d2);
 
end







