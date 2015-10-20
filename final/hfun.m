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
perfi = 5000;
d0 = 1;
alpht = 2.5;
gama = 2;
v = normrnd(1,0.01);
l = x(3:4,0);
e = normrnd(0,0.01);
N = 264;
r(:,1) = [-250,-400];
for j = 1:N
	r(1,j) = r(1,1) + mod(j, 22) * 35;
	r(2,j) = r(2,1) + (j / 22) * 41;
end;

for i = 1:N,	
	y(i) = (perfi .* d0 .^ alpht) / (distance(r(:,i) - l)) .^ alpht + v;
	if y(i) > gama
		z(i) = 20 + e;
	else
		z(i) = e;
	end;
end;
y = z;




