% kalgen.m:  Generates nominal matrices for Kalman filter problem
% based on user inputs.  Creates H, R, Q, I, and F.
% R should be mxm, H should be mxn, and Q should be nxn and F should be nxn.

clear;

n=input('Enter number of dynamic states: ');
m=input('Enter number of measurements (or dimension of measurement vector): ');
proc_noise = input('Enter value for typical process noise variance: ' );
meas_noise = input('Enter value for typical measurement noise variance: ' );

I =  eye(n);
P0 = proc_noise * eye(n);
Q =  proc_noise * eye(n);
R =  meas_noise * eye(m);

% Observation Geometry (Measurment) Matrix:
% This creates an upper diagonal matrix of 1's.
for i=1:m
    for j=1:n
        if i<=j
            H(i,j) = 1;
        else
            H(i,j) = 0;
        end
    end
end

% State Dynamics Matrix:
% Creates a standard companion matrix for a differential
% equation with constant coefficients.
for i=1:n
    for j=1:n
        if (j==i+1)
            F(i,j) = 1;
        else
            F(i,j) = 0;
        end
    end
end
% Build last row of companion matrix from user inputs:
for j=1:n
    prompt=['Enter coeff ',num2str(j-1),' from diff. eq.: '];
    F(n,j)=input(prompt);
    F(n,j)=-F(n,j);
end

display('Basic matrices have been created.');
display('Modify them manually to suit your application.');
display('Once you have finalized your matrices, save them');
display('to a .mat file, so that you can re-load these values');
display('at a later time.  Otherwise, you are ready to run KCAT.');
