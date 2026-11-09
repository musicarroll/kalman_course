% kf resistor demo
% Demonstrates 1-state model but using measurement and process noise,
% and Kalman gain.  Full-up 1-state Kalman filter.

% Demo ideas:  1) Initial estimate under;  2) Initial estimate over;
% 3) large sigma;  4)  small sigma;  5) large sample size;  6) small sample
% size
num_samples = 30;
truth = 220.6;
%z = truth + mysigma*randn(1,num_samples);
% Check to see if measurements are already in workspace (from runavgdemo.m)
exist meas;
if ~ans
    exist mysigma;
    if ~ans
        mysigma = 0.5;
        z = truth + mysigma*randn(1,num_samples);
    else
        z = truth + mysigma*randn(1,num_samples);
    end
else
    z = meas;
end


%Kalman-specific parameters:
Phi = 1;
Q = 5.0;
P0 = 1.0;
R=0.0005^2;
H=1;

x_true = truth*ones(1,num_samples);
x_hat = zeros(1,num_samples);
x_hat(1)=225;
P = zeros(1,num_samples);
pre = zeros(1,num_samples);
post = zeros(1,num_samples);
P(1)=P0;
post(1) = P(1);
pre(1) = P(1);
for k=2:num_samples
    prior_est = x_hat(k-1); % No dynamics!
    prior_P = P(k-1) + Q;
    pre(k) = prior_P;
    K = prior_P*H'*inv(H*prior_P*H' + R);
    x_hat(k) = prior_est + K*(z(k)-H*prior_est);
    P(k) = (1-K*H)*prior_P*(1-K*H)' + K*R*K';
    post(k) = P(k);
end
figure(1)
plot(1:num_samples,x_true,'b--*',1:num_samples,z,'r--o',1:num_samples,x_hat,'d:k');
legend('Truth','Measurements','Estimates','Location','southeast');
xlabel('Time')
ylabel('Resistance (Ohms)')
title('Kalman Estimator: Resistor')
display('Error: ')
display(x_hat(num_samples)-truth)

% To get a sawtooth plot, you can use the attached function I wrote called collate.m.
% If you have two vectors of variances, v1, v2, and one is supposed to be before measurement update and one after, you create
% a new vector twice in length by v=collate(v1,v2).   Then, if t is a the time vector for the period in question, you collate it with itself by
%new_time = collate(t,t).   Then you can plot v against new_time:   plot(new_time, v).

collated = collate(pre,post);
t=1:num_samples;
new_time = collate(t,t);
figure(2)
plot(new_time,collated);
title('Variance in Resistance (without Process Noise)')
xlabel('Time')
ylabel('Variance (Ohms^2)');



