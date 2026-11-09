% damposc.m  Damped harmonic oscillator
% See Harmonic Oscillator examtple in Day 1, Segment 2
% 
clear
num_samples = 30;

delta_t = 1;
m = 1;    % kg
c = 0.1;  % netwon/meter  Spring constant
b = 0.1; % newton sec / meter  dampling coefficient

Phi = expm([0 1; -b/m -c/m]*delta_t);

Q = [0.00001 0; 0 0.000001];
%Q = zeros(2);
P0 = [0.01 0; 0 0.01];
R=0.1^2;
H=[1 0];
w1 = sqrt(Q(1,1))*randn(1,num_samples);
w2 = sqrt(Q(2,2))*randn(1,num_samples);

w = [w1;w2];
x_true = zeros(2,num_samples);
x_true(:,1) = [1;0] + w(:,1);

for k=2:num_samples
    x_true(:,k) = Phi*x_true(:,k-1) + w(:,k-1);
end


x_hat = zeros(2,num_samples);
prior_est = zeros(2,num_samples);
x_hat(:,1)= [1;0];

P = zeros(2,2,num_samples);
prior_P = zeros(2,2,num_samples);

pre = zeros(1,num_samples);
post = zeros(1,num_samples);
z = zeros(1,num_samples);

P(:,:,1)=P0;
post(1) = P(1,1,1);
pre(1) = P(1,1,1);
z(1) = x_true(1,1)+sqrt(R)*randn();

for k=2:num_samples
    prior_est(:,k) = Phi*x_hat(:,k-1); % No dynamics and noise noise because this is an estimate!
    prior_P(:,:,k) = Phi*P(:,:,k-1)*Phi' + Q;
    pre(k) = prior_P(1,1,k);
    K = prior_P(:,:,k)*H'*inv(H*prior_P(:,:,k)*H' + R);
    z(k) = x_true(1,k)+sqrt(R)*randn();
    x_hat(:,k) = prior_est(:,k) + K*(z(k)-H*prior_est(:,k));

    P(:,:,k) = (eye(2)-K*H)*prior_P(:,:,k)*(eye(2)-K*H)' + K*R*K';
    post(k) = P(1,1,k);
end

% RMS Errors:

rms_pos = sqrt(mean((x_true(1,:)-x_hat(1,:)).^2));
rms_vel = sqrt(mean((x_true(2,:)-x_hat(2,:)).^2));

figure(1)
subplot(2,1,1), plot(1:num_samples,x_true(1,:),'b--*',1:num_samples,z,'r--o',1:num_samples,x_hat(1,:),'d:k');
legend('Truth','Measurements','Estimate','Location','southeast');
xlabel('Time')
ylabel('Position (m)')
title('Kalman Estimator: Damped Harmonic Oscillator')
str = {['Meas Noise Variance: ',num2str(R)],['RMS Pos Err: ',num2str(rms_pos)]};
annotation('textbox', [0.7,0.8,0.15,0.05],...
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




