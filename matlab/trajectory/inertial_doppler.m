% inertial_doppler.m
% This script runs the KCAT simulation using a simple 5-d (one deg of freedom)
% inertial/doppler model.
% This model was taken from Savage pg 426.
% States:
% 1 = earth rate plus gyro bias
% 2 = INS platform tilt
% 3 = INS velocity error
% 4 = INS position error
% 5 = Doppler scale factor error

% First clear all arrays:
clear;

% Second, load harmosc.mat model parameters.
load('five-d-inertial-doppler.mat');

% Run the simulation:
[t,x,Cov1,Cov2,Kalgain,Phi] = sim('kcat');

% Plot the results.

figure;
len=length(t);
for i=1:len
% Diagonal elements of covariance matrix prior to update:
    vec1(i) = Cov1(1,1,i);
    vec2(i) = Cov1(2,2,i);
    vec3(i) = Cov1(3,3,i);
% Diagonal elements of covariance matrix after update:
    vec6(i) = Cov2(1,1,i);
    vec7(i) = Cov2(2,2,i);
    vec8(i) = Cov2(3,3,i);
% Kalman gains:
    k1(i)=Kalgain(1,1,i);
    k2(i)=Kalgain(2,1,i);
% Diagonal elements of Transition matrix:
    phi1(i) = Phi(1,1,i);
    phi2(i) = Phi(2,2,i);
end
subplot(2,1,1), plot(t,vec1,t,vec6);
legend('P(1,1)(-)','P(1,1)(+)')
title('KCAT Simulation. Diagonal Elements of Error Covariance Matrix (before and after updates)\newline 5-State Inertial Doppler Model');
subplot(2,1,2), plot(t,vec2,t,vec7);
legend('P(2,2)(-)','P(2,2)(+)');

figure;
plot(t,k1,t,k2);
title('KCAT Simulation.  Kalman Gains:  5-State Inertial Doppler Model');
legend('Gyro Bias Gain','Tilt Gain');

figure;
plot(t,phi1,t,phi2);
title('KCAT Simulation. Diagonal Elements of Transition Matrix:  5-State Inertial Doppler Model');
legend('\Phi(1,1)','\Phi(2,2)');

% Splice covariance:

[time,vec16] = interleave(t,vec1,vec6);
figure;
initSegLen = floor(.15*length(time));
plot(time(1:initSegLen),vec16(1:initSegLen));
