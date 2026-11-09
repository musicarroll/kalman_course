function X = solve_ut(U,Y)
%
% File Name:  solve_ut.m
%
%
% Related Programs and Documents:  compute_ud.m
%
%
% Run-time Environment Constraints:  Must run in Matlab or Simulink.
%
%
%
%    DESCRIPTION:
% X = solve_ut(U,Y)
% Solves the matrix / vector equation
% UX=Y for X, where U is unit upper triangular, 
% and Y is conformal to U (i.e. #cols(U)=#rows(Y)).
% To use this function iteratively, call it with
% Y=solve_ut(U,Y) repeatedly to update Y.
%
% X = solve_ut(U) with U rectangular (non-square) works as well.
% In this case, U is split into a new U and Y with U being sqare and
% having exactly as many columns as the old U has rows.  Y contains as
% many columns as are left in the old U after splitting off U.
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
    [m,n]=size(U);
    [q,p]=size(Y);

% Verify that U is square and Y has the same 
% number of rows as U has columns.
    if m~=n
        X=[];
        error('m different from n, i.e. U is not square.');
        return;
    end
    if m~=q 
        X=[];
        error('m different from q, i.e., Y is not conformal to U.');
        return;
    end
elseif (nargin == 1)
% U cannot be square in this case.  In fact, n should be > m    
    [m,n] = size(U);
    if (n <= m)
        error('U is either square or has fewer columns than rows');
        X=[];
        return;
    end
    Utmp = U(1:m,1:m);
    Y    = U(1:m,m+1:n);
    U    = Utmp;
    p = n-m;
else
    X=[];
    error('solve_ut needs to have one or two arguments.');
end


% Verify that U is unit upper triangular.
% Only two things can cause a matrix not to be 
% unit upper triangular: either it has something
% other than 1 on the main diagonal, or it has
% something other than 0 below the main diagonal
for i=1:m
    for j=1:m
        if i==j
            if U(i,j) ~= 1
                X=[];
                error('U has non-unit on diagonal');
                return;
            end
        elseif i>j
            if U(i,j) ~= 0
                X=[];
                error('U has non-zero element below diagonal');
                return;
            end
        end
    end
end


for j=1:p
    for i=m:-1:1
        X(i,j)=Y(i,j);
        for k=i+1:m
            X(i,j)=X(i,j)-U(i,k)*X(k,j);
        end
    end
end
return;


%
%
%    $History: template.m $
%

