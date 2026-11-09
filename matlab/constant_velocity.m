% constant_velocity.m  Simple 1d uniform motion
% See Non-acelerated Motion examtple in Day 1, Segment 2
% 
clear
num_samples = 100;

delta_t = 1;

% State Transition Matrix:
Phi = [1 delta_t; 0 1];

% Process Noise:
Q = [0.0001 0; 0 0.000001];
%Q = zeros(2);
% Process noise sample trajectory:
w1 = sqrt(Q(1,1))*randn(1,num_samples);
w2 = sqrt(Q(2,2))*randn(1,num_samples);
w = [w1;w2];

% Initial estimation uncertainty (covarience):
P0 = [0.01 0; 0 0.01];

% Measurement noise covariance
R=1^2;
% Meaurement matrix:  Position measurement only
H=[1 0];

% Truth model
x_true = zeros(2,num_samples);
% Initial truth:
x_true(:,1) = [0;1] + w(:,1);
% Apply Phi to truth and add process noise:
for k=2:num_samples
    x_true(:,k) = Phi*x_true(:,k-1) + w(:,k-1);
end

% Estimate:
x_hat = zeros(2,num_samples);
prior_est = zeros(2,num_samples); % (x-)
% Initial state:
x_hat(:,1)= [0;1];

% Estimation Error Covariance:
P = zeros(2,2,num_samples);
prior_P = zeros(2,2,num_samples);  % (P-)

% Scalar sequence to use for sawtooth plotting:
pre = zeros(1,num_samples); 
post = zeros(1,num_samples);
z = zeros(1,num_samples);

% Initialize variables for the loop
P(:,:,1)=P0;
post(1) = P(1,1,1);
pre(1) = P(1,1,1);

% Initialize measurement sequence (= truth + meas noise):
z(1) = x_true(1,1)+sqrt(R)*randn();

% The Kalman loop:
for k=2:num_samples
    % Extrapolation:
    prior_est(:,k) = Phi*x_hat(:,k-1); % No dynamics and noise noise because this is an estimate!
    prior_P(:,:,k) = Phi*P(:,:,k-1)*Phi' + Q;
    pre(k) = prior_P(1,1,k);
    % Kalman gain:
    K = prior_P(:,:,k)*H'*inv(H*prior_P(:,:,k)*H' + R);
    % The current measurement:
    z(k) = x_true(1,k)+sqrt(R)*randn();
    % Update:
    x_hat(:,k) = prior_est(:,k) + K*(z(k)-H*prior_est(:,k));
    P(:,:,k) = (eye(2)-K*H)*prior_P(:,:,k)*(eye(2)-K*H)' + K*R*K';
    post(k) = P(1,1,k);
end


% Plot truth, measuremsnts and estimates.

%RMS Errors:
rms_pos = sqrt(mean((x_true(1,:)-x_hat(1,:)).^2));
rms_vel = sqrt(mean((x_true(2,:)-x_hat(2,:)).^2));

figure(1)
subplot(2,1,1), plot(1:num_samples,x_true(1,:),'b--*',1:num_samples,z,'r--o',1:num_samples,x_hat(1,:),'d:k');
legend('Truth','Measurements','Estimate','Location','southeast');
xlabel('Time')
ylabel('Position (m)')
title('Kalman Estimator: Non-accelerated 1d Motion')
str = {['Meas Noise Variance: ',num2str(R)],['RMS Pos Err: ',num2str(rms_pos)]};
annotation('textbox', [0.6,0.6,0.15,0.05],...
           'String', str);
subplot(2,1,2), plot(1:num_samples,x_true(2,:),'b--*',1:num_samples,x_hat(2,:),'d:k');       
legend('Truth','Estimate','Location','southeast');
xlabel('Time (s)')
ylabel('Velocity (m/sec)')
str = {['RMS Vel Err: ',num2str(rms_vel)]};
annotation('textbox', [0.7,0.4,0.15,0.03],...
           'String', str);

display('Error: ')
display(x_hat(:,num_samples)-x_true(:,num_samples))

% To get a sawtooth plot, you can use the attached function I wrote called collate.m.
% If you have two vectors of variances, v1, v2, and one is supposed to be before measurement update and one after, you create
% a new vector twice in length by v=collate(v1,v2).   Then, if t is a the time vector for the period in question, you collate it with itself by
%new_time = collate(t,t).   Then you can plot v against new_time:   plot(new_time, v).

collated = collate(pre,post);
t=1:num_samples;
new_time = collate(t,t);
figure(2)
plot(new_time,collated);
title('Variance in Position (with Process Noise)')
xlabel('Time')
ylabel('Variance (m^2)');
