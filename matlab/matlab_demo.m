% Matlab demo
% Vectors
%   Row vector:
v = [1 2 3]; % Or separate columns with comma: v = [1, 2, 3];
%   Column vector:
u = [1;2;3]; % Note: Rows separated with semicolon
% Natural multiplication:
x=v*u;
%Displaying values:
v
u
x
% Matrices: Combinations of rows and columns:
A = [1 2 3; 4 5 6; 7 8 9];
w = A*u;
w
% Matrix operators like inverse and transpose:
B = inv(A);
C = A';
D = u*u';
B
C
D
t=0:0.1:1;
t

