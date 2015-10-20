function [x,q] = qmcmultiresample(sample,q,N,b,Neff,Nth)
% sample from [a(t),b(t)] and base is d;
% exploit Halton sequence to generate samples;
x = zeros(1,N);
xpf = sample * q' ;
% u = sort(rand(N,1));
u = zeros(1,N);
u(1,1) = (1/N)*rand(1);
Q = cumsum(q);
i = 1;
j = 1;
index = zeros(N,1);

% 依据重采样方法中最常用的随机采样方法
while  j <= N,       
    u(1,j) = u(1,1)+(1/N)*(j-1);
    while   i <= N & u(1,j) > Q(1,i),             % i <= N 预防下标溢出，两个条件不可颠倒
%           x(i,1) = xPre(j,1);
          i = i+1;
    end;
    index(i,1)=index(i,1)+1;
    j = j+1;   
end;

M = find(index(:,1)>0) ;
[row,col]= size(M);
num = sum(index(:,1,1));
% L = (max(sample(1,M))-min(sample(1,M)))*(1-Neff/N)/row; 
%L=0;

for i =1:row,
   n = index(M(i,1),1) ;
   e = sample(1,M(i,1));  
    if i>1
       strt = sum(index(1:M(i-1,1),1))+1;
   else
       strt = 1;
   end;
   stp = sum(index(1:M(i,1),1));
   q(1,strt:stp) = q(1,M(i,1))/n ;
   if n <= 2
       x(1,strt:stp)=e;
   else
%    l = L/(N*sqrt(n));     
   x(1,strt:stp-1) = normrnd(e,Neff/N,n-1,1);
%    x(1,stp-1)=e;
   x(1,stp)=e*n - sum(x(strt:stp-1));   
   end;
end;
% if N>stp
% x(1,stp+1:N)=qmcinterval(xpf-0.5*L,xpf+0.5*L,b,N-stp);
% end;
