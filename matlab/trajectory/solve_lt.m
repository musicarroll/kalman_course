function X = solve_lt(L,Y)
%
% File Name:  solve_lt.m
%
%
% Related Programs and Documents:  compute_ud.m, solve_ut.m
%
%
% Run-time Environment Constraints:  Must run in Matlab or Simulink.
%
%
%
%    DESCRIPTION:
% X = solve_lt(L,Y)
% Solves the matrix / vector equation
% LX=Y for X, where L is unit lower triangular, 
% and Y is conformal to L (i.e. #cols(L)=#rows(Y)).
% To use this function iteratively, call it with
% Y=solve_lt(L,Y) repeatedly to update Y.
%
% X = solve_lt(L) with L rectangular (non-square) works as well.
% In this case, L is split into a new L and Y with L being sqare and
% having exactly as many columns as the old L has rows.  Y contains as
% many columns as are left in the old L after splitting off U.
%
%
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

% First, check to make sure we have the right
% number of arguments to this function.
if (nargin == 2)
    [m,n]=size(L);
    [q,p]=size(Y);

% Verify that L is square and Y has the same 
% number of rows as L has columns.
    if m~=n
        X=[];
        error('m different from n, i.e. L is not square.');
        return;
    end
    if m~=q 
        X=[];
        error('m different from q, i.e., Y is not conformal to L.');
        return;
    end
elseif (nargin == 1)
% L cannot be square in this case.  In fact, n should be > m    
    [m,n] = size(L);
    if (n <= m)
        error('L is either square or has fewer columns than rows');
        X=[];
        return;
    end
    Ltmp = L(1:m,1:m);
    Y    = L(1:m,m+1:n);
    L    = Ltmp;
    p = n-m;
else
    X=[];
    error('solve_lt needs to have one or two arguments.');
end


% Verify that L is unit lower triangular.
% Only two things can cause a matrix not to be 
% unit lower triangular: either it has something
% other than 1 on the main diagonal, or it has
% something other than 0 above the main diagonal
for i=1:m
    for j=1:m
        if i==j
            if L(i,j) ~= 1
                X=[];
                error('L has non-unit on diagonal');
                return;
            end
        elseif i<j
            if L(i,j) ~= 0
                X=[];
                error('L has non-zero element below diagonal');
                return;
            end
        end
    end
end


for j=1:p
    for i=1:m
        X(i,j)=Y(i,j);
        for k=i-1:-1:1
            X(i,j)=X(i,j)-L(i,k)*X(k,j);
        end
    end
end
return;


%
%
%    $History: template.m $
%

