function [Uinv] = inv_u(U)

if (nargin ~= 1)
    U=[];
    error('inv_u needs one argument.');
end

% Verify that U is square.

[m,n]=size(U);

if m~=n
    U=[];
    error('U is not square.');
    return;
end

% Verify that U is unit upper triangular:
for i=m:-1:1
    if U(i,i) ~= 1
        error('U does not have a unit diagonal.');
    end
    for j=i-1:-1:1
        if U(i,j) ~=0
            error('U does not have zeros below the diagonal.');
        end
    end
end
Uinv=U;
for i=m:-1:1
    for j=m:-1:i+1
        Uinv(i,j)=-Uinv(i,j);
        for k=i+1:j-1
            Uinv(i,j)=Uinv(i,j)-Uinv(i,k)*Uinv(k,j);
        end
    end
end
return;



%
%
%    $History: inv_u.m $
%
