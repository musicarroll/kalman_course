% kal2ndord.m:  Generates matrices for second order harmonic oscillator Kalman filter problem
% based on user inputs.  Creates H, R, Q, and F.
% R should be mxm, H should be mxn, and Q should be nxn and F should be nxn
% Calling 'kcat' via [t,x,Pm,Pp,Kout] = sim('kcat') yields the extrapolated covariance matrix Pm 
% prior to update, the updated covariance matrix Pp after update, and the Kalman gain Kout
% for each cycle.

clear;

n=2;
m=1;
proc_noise = input('Enter value for process noise variance: ' );
meas_noise = input('Enter value for measurement noise variance: ' );

I =  eye(n);
disp('The initial process noise covariance ');
disp('will be generated as a diagonal matrix using ');
disp('the process noise you specified.  However, to match ');
disp('parameters in LeMay pg. 25, you need to modify P0 as');
[100 10; 10 50]
disp('and Q as');
[0 0;0 5]
disp('Similarly, the R matrix in LeMay is');
[150]
P0 = proc_noise * eye(n);
Q =  proc_noise * eye(n);
Q(1,1)=0;
R =  meas_noise * eye(m);

% Observation Geometry Matrix:

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

for i=1:n
    for j=1:n
        if (j==i+1)
            F(i,j) = 1;
        else
            F(i,j) = 0;
        end
    end
end
% Get last row of companion matrix:
    disp('To determine the model, you will need to specify the ');
    disp('undamped natural frequency and the damping coefficient.');
    disp('The model in J. LeMay, pg 19 uses omega=2 rad/sec and zeta=0.5');
    disp('The time step in the LeMay model is 0.2 sec. The default in ');
    disp('KCAT is 1 sec.  You will have to change that value in KCAT if ');
    disp('you so desire.');
    prompt=['Enter omega (natural freqency): '];
    omega=input(prompt);
    F(2,1)=-omega^2;
    prompt=['Enter zeta (damping coefficient): '];
    zeta=input(prompt);
    F(2,2)=-2*zeta*omega;
Phi = [1 1;-omega^2 1-2*zeta*omega];