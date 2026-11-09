% kf_resistor_demo.m
% Matlab script to demonstrate a simple, static, 1-state KF model using both measurement and process noise,
% and Kalman gain.  Full-up 1-state Kalman filter, but trivial dynamics (constant dynamics = static).

% Demo Suggestions:  Try running with 1) Initial estimates too low;  2) Initial estimate too high;
% 3) large sigmas;  4)  small sigmas;  5) large sample size;  6) small sample size

num_samples = 30;
truth = 220.6;  % static state model

%Kalman-specific parameters:
Phi = 1;    % Constant dynamics
Q = 5.0;    % Process noise covariance (1x1 matrix)
P0 = 1.0;   % Initial estimation error covariance (1x1 matrix)
R=0.0005^2; % Initial measurement noise covariance (1x1 matrix)
% Measurement z = truth + mysigma*randn(1,num_samples);
z = truth + sqrt(R)*randn(1,num_samples);
H=1;        % Measurement matrix (1x1) 

x_true = truth*ones(1,num_samples);  % True state
x_hat = zeros(1,num_samples);        % Data series to hold state estimates
x_hat(1)=225;                        % Initial state estimate / guess
P = zeros(1,num_samples);            % Estimation error covariance matrix
P(1)=P0;                             % Initial covariance
pre = zeros(1,num_samples);          % Prior covariance vector for plotting
post = zeros(1,num_samples);         % Posterior covariance for plotting
post(1) = P(1);                      % Initial pre value
pre(1) = P(1);                       % Initial post value

% Main Kalman loop:

for k=2:num_samples              % Start at k=2, because we already have k=1 as initial values
    prior_est = x_hat(k-1);      % Static problem; no dynamics!
    prior_P = P(k-1) + Q;        % Temporary prior P:  Simply add process noise, since there are no dynamics
    pre(k) = prior_P;            % Set pre covariance for this k
    K = prior_P*H'*inv(H*prior_P*H' + R);  % Compute Kalman gain
    x_hat(k) = prior_est + K*(z(k)-H*prior_est);  % State update using prior estimate, measurement and Kalman gain
    P(k) = (1-K*H)*prior_P*(1-K*H)' + K*R*K';     % Covariance update
    post(k) = P(k);              % Set posterior vector for plotting
end

% Plot results:

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



