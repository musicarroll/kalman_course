function [Minv] = ud_inv(M)

% local definitions
%	none	<or definitions if there are any>

% default settings
%	none	<or settings>

% external functions

% external string

% internal functions	<list of functions in this file>

% global data

% local data

% public functions

% private functions

if (nargin ~= 1)
    U=[];
    error('ud_inv needs one argument.');
end

% Verify that M is square.

[m,n]=size(M);

if m~=n
    U=[];
    error('M is not square.');
    return;
end
% First, get UD factors:
[U,D] = compute_ud(M);
% Second, invert U:
Uinv  = inv_u(U);
% 3) Invert D:
for i=1:m
    Dinv(i,i) = 1/D(i,i);
end
Minv  = Uinv'*Dinv*Uinv;
return;

