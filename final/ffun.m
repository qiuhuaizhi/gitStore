function [y] = ffun(x,t);
% PURPOSE : Process model function.
% INPUTS  : - x:  The evaluation point in the domain.
% OUTPUTS : - y: The value of the function at x.
% AUTHORS  : 
% DATE     : 


if nargin < 2, error('Not enough input arguments.'); end

%beta = 0.5;                 % Autoregressive parameter.
%y = 1 + sin(4e-2*pi*t) + beta*x; 

Cu = diag([0.05,0.01],0);
mu = [0,0];
ut = mvnrnd(mu,Cu,1);

Gx = diag(ones(1,4),0) + diag([Ts,Ts],2);

Gu = [(Ts ^ 2) / 2, 0; 0, (Ts ^ 2) / 2; Ts, 0; 0, Ts]

y = Gx * x + Gu * ut;





