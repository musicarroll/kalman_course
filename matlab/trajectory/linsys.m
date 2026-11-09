% Direct methods for solving simultaneous linear 
% equations

A=[1 -2 1;2 1 0; 0 0 1];
b = [0;1;-1];
A=input('Enter A matrix: ');
b = input('Enter known b vector: ');
disp('A is');
disp(A);
disp('Determinant of A is: ');
disp(det(A));
if det(A)~=0
    x=A\b;
    disp('Solution is: ');
    x
else
    disp('Cannot be solved because A is singular.');
end
