function [roll,pitch,yaw,alpha,Cbn] = getInitAttitude(roll0,pitch0,yaw0);
% This function initializes the attitude
% parameters of roll, pitch and yaw, and the DCM
% connecting this rpy to local level coordinates.
roll  = roll0;
pitch = pitch0;
yaw   = yaw0;
alpha = 0;  % Initial wander angle = 0
Cbn   = euler2DCM(roll,pitch,yaw);
return;
