function [U,D] = compute_ud(P)

% First, check to make sure we have the right
% number of arguments to this function.
if (nargin ~= 1)
    U=[];
    D=[];
    error('compute_UD needs one argument.');
end

% Verify that P is square.

[m,n]=size(P);

if m~=n
    U=[];
    D=[];
    error('P is not square.');
    return;
end
ok=isposdef(P);
if ~ok
    error('P is not positive definite.');
end

% Input is fine, so now compute the UD factors:
for j=m:-1:1
    for i=j:-1:1
        covar=P(i,j);
        for k=j+1:m
            covar=covar-U(i,k)*D(k,k)*U(j,k);
        end
        if i==j
            D(j,j)=covar;
            U(j,j)=1;
        else
            U(i,j)=covar/D(j,j);
        end
    end
end

return;


%
%
%    $History: template.m $
%

