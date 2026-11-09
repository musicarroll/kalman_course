% kalgen.m:  Generates nominal matrices for continuous Kalman filter problem
% based on user inputs.  Creates H, R, Q, I, and F.  Also creates state
% transition matrix Phi for diescrete steps of length delta_t, starting at
% t=0.
% R should be mxm, H should be mxn, and Q should be nxn and F should be nxn.
% No control input is used in this model.

clear;

display('*** Building State Dynamics Model ***')
display('for a LTI system.');
display(' ');
n=input('Enter number of dynamic states: ');
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
display(' ');
display('Now specify coefficients of differential equation,')
display('starting with the coefficient of D^(n-1).');
display('(Note: Highest order coeff of D^n assumed = 1 ...)');
for j=1:n
    prompt=['Enter coeff of D^(n-',num2str(j),') from diff. eq.: '];
    F(n,n-j+1)=-input(prompt);
end
display(' ');
display('*** Building state transition matrix for discrete sampling periods ***')
delta_t=input('Enter delta T (Kalman cycle period): ');
% State transition matrix:
Phi = expm(F*delta_t);

I =  eye(n);

display(' ');
init_P_diag = input('Enter value for initial estimation error variance (P0): ' );
P0 = init_P_diag * I;

display(' ');
proc_noise = input('Enter value for typical process noise variance (Q): ' );
Q =  proc_noise * I;

display(' ');
display('*** Building Measurement Model ***');
display(' ');
m=input('Enter number of measurements (i.e., dimension of measurement vector): ');
meas_noise = input('Enter value for typical measurement noise variance (R): ' );

R =  meas_noise * eye(m);
R0 = R;

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

display(' ');
display('*** Basic matrices have been created. ***');
display('Modify them manually to suit your application.');
display('Typically, the Q and R matrices will need to be modified as');
display('they are initially set to have the same nominal values on the');
display('the diagonal.');
display('Once you have finalized your matrices, save them');
display('to a .mat file, so that you can re-load these values');
display('at a later time.  Otherwise, you are ready to run covariance_analysis.m');
display(' ');

