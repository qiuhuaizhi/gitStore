function [x]=resample(q,xPre,N)
u = sort(rand(N,1));
Q = cumsum(q);
i = 1;
j = 1;
index = zeros(N,1);
% �����ز�����������õ������������
while  j <= N,                         
    if   i <= N & Q(j,1) > u(i,1),             % i <= N Ԥ���±�����������������ɵߵ�
%           x(i,1) = xPre(j,1);
          i = i+1;
    else
        if j>1
          index(j)=i-sum(index(1:j-1));
        else
            index(j)=i;
        end;
          j = j+1;
    end;
end;
x=index;