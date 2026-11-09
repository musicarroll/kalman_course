% Covariance analysis of C-MIGITS II/III Kalman Filter
%
% Author:  M. Carroll
% Revision History
% Date      Author      Change
% ----      ------      ------
% 03/13/02  MLC         Initial Release
%
% Purpose
%
clf;
disp('Covariance analysis of C-MIGITS II Kalman Filter,');
disp(' ');
disp('Plots histories of 28 mean squared state');
disp('uncertainties and 28 magnitudes of Kalman gains');
disp('for intersample intervals of 5, 10 and 15 seconds.');
disp(' ');
disp('28 state variables:');
disp('  1. x position error');
disp('  2. y position error');
disp('  3. z position error');
disp('  4. x velocity error');
disp('  5. y velocity error');
disp('  6. z velocity error');
disp('  7. x tilt');
disp('  8. y tilt');
disp('  9. azimuth error');
disp(' 10. reference heading error');
disp(' 11. GPS freq rate error');
disp(' 12. GPS freq error');
disp(' 13. GPS range bias error');
disp(' 14. x position changes');
disp(' 15. y position changes');
disp(' 16. z position changes');
disp(' 17. x gyro scale factor');
disp(' 18. y gyro scale factor');
disp(' 19. z gyro scale factor');
disp(' 20. x accel scale factor');
disp(' 21. y accel scale factor');
disp(' 22. z accel scale factor');
disp(' 23. x gyro bias');
disp(' 24. y gyro bias');
disp(' 25. z gyro bias');
disp(' 26. x accel bias');
disp(' 27. y accel bias');
disp(' 28. z accel bias');
%  Contstants used in this model
sigma1sq = (103/3)^2;
sigma2sq = 1.3E-8;
sigmarsq = (1000)^2;
sigmatsq = (.017)^2;
%big_omega_x = 
rho      = 0.5; % relative rate
g        = 9.80665; % m/sec/sec
earth_radius = 6.378E6;
omega_x = .01 % rad/sec
omega_y = .01 % rad/sec

%
% State Transition matrix (the part not depending on T)
%
Phi      = eye(28);
Q        = eye(28);
Q(1,1)   = sigma2sq;
Q(3,3)   = sigma1sq;
Q(6,6)   = sigma2sq;
Q(1,4)   = .005;  % assume some coupling between x pos and x vel
Q(4,1)   = .005;  % assume some coupling between x pos and x vel
R        = zeros(1,1);
R(1,1)   = sigmarsq;
H        = zeros(1,28);
H(1,1)   = 1;
H(1,2)   = 1;
H(1,3)   = 1;
H(1,4)   = 1;
%
% arrays for saving data to be plotted
%
t        = zeros(3,32); % time (for 3 plots)
pos_x_cov     = zeros(3,32); % x pos covariance
pos_y_cov    = zeros(3,32); % y position covariance
pos_z_cov     = zeros(3,32); % Bearing covariance
vel_x_cov    = zeros(3,32); % x vel covariance
vel_y_cov   = zeros(3,32); % y position Noise covariance
vel_z_cov   = zeros(3,32); % x vel Noise covariance
pos_x_kal_gain      = zeros(3,32); % x position Kalman gain
pos_y_kal_gain     = zeros(3,32); % y position Kalman gain
pos_z_kal_gain      = zeros(3,32); % z position Kalman gain
vel_x_kal_gain     = zeros(3,32); % x velocity Kalman gain
vel_y_kal_gain    = zeros(3,32); % y velocity Kalman gain
vel_z_kal_gain    = zeros(3,32); % z velocity Kalman gain
N=0;
   for T = 5:5:15,
   N=N+1;
   disp(['Simulating C-MIGITS III at ',num2str(T),' second intervals.']);
   Phi(1,2) = T;
   Phi(2,3) = -g*T;
   Phi(3,2) = T/earth_radius;
   
   P        = zeros(28);
   P(1,1)   = sigmarsq;
   P(1,2)   = sigmarsq/T;
   P(2,1)   = P(1,2);
   P(2,2)   = 2*sigmarsq/T^2 + sigma1sq;
   P(3,3)   = sigma1sq;
   P(4,4)   = sigmatsq;
   P(4,5)   = sigmatsq/T;
   P(5,4)   = P(4,5);
   P(5,5)   = 2*sigmatsq/T^2 + sigma2sq;
   P(6,6)   = sigma2sq;
      for cycle=0:15,
%
% Save a priori values
%
      prior      = 2*cycle+1;
      t(N,prior) = T*cycle;
      K          = P*H'/(H*P*H'+R); % compute the Kalman gain
      pos_x_cov(N,prior)   = P(1,1); % x position covariance
      pos_y_cov(N,prior)  = P(2,2); % y position covariance
      pos_z_cov(N,prior)   = P(4,4); % z position covariance
      vel_x_cov(N,prior)  = P(5,5); % x velocity covariance
      vel_y_cov(N,prior) = P(3,3); % y velocity covariance
      vel_z_cov(N,prior) = P(6,6); % z velocity covariance
      pos_x_kal_gain(N,prior)    = sqrt(K(1,1)^2); 
      pos_y_kal_gain(N,prior)   = sqrt(K(2,1)^2); 
      pos_z_kal_gain(N,prior)    = sqrt(K(4,1)^2); 
      vel_x_kal_gain(N,prior)   = sqrt(K(5,1)^2); 
      vel_y_kal_gain(N,prior)  = sqrt(K(3,1)^2); 
      vel_z_kal_gain(N,prior)  = sqrt(K(6,1)^2); 
%
% Save a posteriori values
%
      P         = P - K*H*P;
      P         = .5*(P+P');
      post      = prior + 1;
      t(N,post) = T*cycle;
      pos_x_cov(N,post)   = P(1,1); % x pos covariance
      pos_y_cov(N,post)  = P(2,2); % y position covariance
      pos_z_cov(N,post)   = P(4,4); % Bearing covariance
      vel_x_cov(N,post)  = P(5,5); % x vel covariance
      vel_y_cov(N,post) = P(3,3); % y position Noise covariance
      vel_z_cov(N,post) = P(6,6); % x vel Noise covariance
      pos_x_kal_gain(N,post)  = sqrt(K(1,1)^2); % x pos Kalman gain
      pos_y_kal_gain(N,post)  = sqrt(K(2,1)^2); % y position Kalman gain
      pos_z_kal_gain(N,post)  = sqrt(K(4,1)^2); % Bearing Kalman gain
      vel_x_kal_gain(N,post)  = sqrt(K(5,1)^2); % x vel Kalman gain
      vel_y_kal_gain(N,post)  = sqrt(K(3,1)^2); % y position Noise Kalman gain
      vel_z_kal_gain(N,post)  = sqrt(K(6,1)^2); % x vel Noise Kalman gain

      P              = Phi*P*Phi' + Q;  % Extrapolate the covariance and add the process noise
      end;
   end;
disp('X, Y Position Covariances and Kalman Gains')
subplot(2,2,1),plot(t(1,:),pos_x_cov(1,:),t(2,:),pos_x_cov(2,:),t(3,:),pos_x_cov(3,:));xlabel('Time (sec)');ylabel('X Pos Cov');
subplot(2,2,2),plot(t(1,:),pos_x_kal_gain(1,:),t(2,:),pos_x_kal_gain(2,:),t(3,:),pos_x_kal_gain(3,:));xlabel('Time (sec)');ylabel('X Pos Gain');
subplot(2,2,3),plot(t(1,:),pos_y_cov(1,:),t(2,:),pos_y_cov(2,:),t(3,:),pos_y_cov(3,:));xlabel('Time (sec)');ylabel('Y Pos Cov');
subplot(2,2,4),plot(t(1,:),pos_y_kal_gain(1,:),t(2,:),pos_y_kal_gain(2,:),t(3,:),pos_y_kal_gain(3,:));xlabel('Time (sec)');ylabel('Y Pos Gain');

disp('Press <ENTER> for plots 13-16.');
pause
disp('Vel y,z Covariances and Pos x,y Gains:  ')
subplot(2,2,1),plot(t(1,:),pos_z_cov(1,:),t(2,:),pos_z_cov(2,:),t(3,:),pos_z_cov(3,:));xlabel('Time (sec)');ylabel('Z Pos Cov');
subplot(2,2,2),plot(t(1,:),pos_z_kal_gain(1,:),t(2,:),pos_z_kal_gain(2,:),t(3,:),pos_z_kal_gain(3,:));xlabel('Time (sec)');ylabel('Z Pos Gain');
subplot(2,2,3),plot(t(1,:),vel_x_cov(1,:),t(2,:),vel_x_cov(2,:),t(3,:),vel_x_cov(3,:));xlabel('Time (sec)');ylabel('X Vel Cov');
subplot(2,2,4),plot(t(1,:),vel_x_kal_gain(1,:),t(2,:),vel_x_kal_gain(2,:),t(3,:),vel_x_kal_gain(3,:));xlabel('Time (sec)');ylabel('X Vel Gain');

disp('Press <ENTER> for plots 17-20.');
pause
disp('Pos z, Vel x,y,z Kalman Gains')
subplot(2,2,1),plot(t(1,:),vel_y_cov(1,:),t(2,:),vel_y_cov(2,:),t(3,:),vel_y_cov(3,:));xlabel('Time (sec)');ylabel('Y Vel Cov');
subplot(2,2,2),plot(t(1,:),vel_y_kal_gain(1,:),t(2,:),vel_y_kal_gain(2,:),t(3,:),vel_y_kal_gain(3,:));xlabel('Time (sec)');ylabel('Y Vel Gain');
subplot(2,2,3),plot(t(1,:),vel_z_cov(1,:),t(2,:),vel_z_cov(2,:),t(3,:),vel_z_cov(3,:));xlabel('Time (sec)');ylabel('Z Vel Cov');
subplot(2,2,4),plot(t(1,:),vel_z_kal_gain(1,:),t(2,:),vel_z_kal_gain(2,:),t(3,:),vel_z_kal_gain(3,:));xlabel('Time (sec)');ylabel('Z Vel Gain');
disp('DONE');

