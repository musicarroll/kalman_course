% harmosc_compare.m
% This script runs the SIMLTIC simulation using two harmonic oscillator models.

% First clear all arrays:
% clear;

% Second, load harmosc.mat model parameters.
%load('harmosc_compare.mat');

% Run the simulation and place results in t and yout.
sim('cov_compare');

% Splice pre and post Covariances:
[tlen,x] = size(yout.time);
for k=2:2:2*tlen
  t(k-1)=yout.time(k/2);
  t(k)=yout.time(k/2);
  cov1(1,k-1)=yout.signals(1).values(1,1,k/2);
  cov1(2,k-1)=yout.signals(1).values(2,2,k/2);
  cov1(1,k)  =yout.signals(2).values(1,1,k/2);
  cov1(2,k)  =yout.signals(2).values(2,2,k/2);

  cov2(1,k-1)=yout.signals(5).values(1,1,k/2);
  cov2(2,k-1)=yout.signals(5).values(2,2,k/2);
  cov2(1,k)  =yout.signals(6).values(1,1,k/2);
  cov2(2,k)  =yout.signals(6).values(2,2,k/2);

  kalgain1_1(k/2) = yout.signals(3).values(1,1,k/2);
  kalgain1_2(k/2) = yout.signals(3).values(2,1,k/2);

  kalgain2_1(k/2) = yout.signals(7).values(1,1,k/2);
  kalgain2_2(k/2) = yout.signals(7).values(2,1,k/2);
end


% Plot the results.

figure;
subplot(2,1,1), plot(t,cov1(1,:),'--+',t,cov2(1,:),'-.o');
title('Comparitive Covariance Analysis of Damped Harmonic Oscillator');
legend(['Position Estimation Error Variance (Model 1) Q1(1,1)=',num2str(Q1(1,1)),'; R1(1,1)=',num2str(R1(1,1))],['Position Estimation Error Variance (Model 2)  Q2(1,1)=',num2str(Q2(1,1)),'; R2(1,1)=',num2str(R2(1,1))]);
subplot(2,1,2), plot(t,cov1(2,:),'--+',t,cov2(2,:),'-.o');
legend(['Velocity Estimation Error Variance (Model 1) Q1(2,2)=',num2str(Q1(2,2)),';  R1(1,1)=',num2str(R1(1,1))],['Velocity Estimation Error Variance (Model 2) Q2(2,2)=',num2str(Q2(2,2)),'; R2(1,1)=',num2str(R2(1,1))]);

figure;
subplot(2,1,1), plot(yout.time,kalgain1_1,'--+',yout.time,kalgain2_1,'-.o');
title('Comparitive Covariance Analysis of Damped Harmonic Oscillator: Gains');
legend('Position Gain from Model 1','Position Gain from Model 2');
subplot(2,1,2), plot(yout.time,kalgain2_1,'--+',yout.time,kalgain2_2,'-.o');
legend('Velocity Gain from Model 1','Velocity Gain from Model 2');
