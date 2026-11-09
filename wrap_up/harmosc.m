% harmosc.m
% This script runs the KCAT simulation using the harmonic oscillator model.

% First clear all arrays:
clear;

% Second, load harmosc.mat model parameters.
load('harmosc.mat');

% Run the simulation:
%[t,x,Cov1,Cov2,Kalgain,Phi] = sim('kcat');
[t,x,Cov1,Cov2,Kalgain,Phi] = sim('simltic');

% Plot the results.

figure;
len=length(t);
% Note: Using element-by-element assignment since vector assignment
% such as vec1 = Cov1(1,1,:) yields a structure of size 1x1x11.
% And such structures are abhorrent to the plot function. :-)
for i=1:len
% Diagonal elements of covariance matrix prior to update:
    vec1(i) = Cov1(1,1,i);
    vec2(i) = Cov1(2,2,i);
% Diagonal elements of covariance matrix after update:
    vec3(i) = Cov2(1,1,i);
    vec4(i) = Cov2(2,2,i);
% Kalman gains:
    k1(i)=Kalgain(1,1,i);
    k2(i)=Kalgain(2,1,i);
% Diagonal elements of Transition matrix:
    phi1(i) = Phi(1,1,i);
    phi2(i) = Phi(2,2,i);
end
subplot(2,1,1), plot(t,vec1,t,vec3)
legend('P(1,1)(-)','P(1,1)(+)')
title('KCAT Simulation. Diagonal Elements of Error Covariance Matrix (before and after updates)\newline 2-State Harmonic Oscillator Model');
subplot(2,1,2), plot(t,vec2,t,vec4);
legend('P(2,2)(-)','P(2,2)(+)');

figure;
plot(t,k1,t,k2);
title('KCAT Simulation.  Kalman Gains:  2-State Harmonic Oscillator Model');
legend('Position Gain','Velocity Gain');

figure;
plot(t,phi1,t,phi2);
title('KCAT Simulation. Diagonal Elements of Transition Matrix:  2-State Harmonic Oscillator Model');
legend('\Phi(1,1)','\Phi(2,2)');

