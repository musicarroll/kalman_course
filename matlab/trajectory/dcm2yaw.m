function [yaw] = dcm2yaw(C,conv);
% dcm2yaw.m is a function that converts the body-to-nav (local level) direction
% cosine matrix to yaw.
yaw = atan2(C(1,2),C(1,1));
if conv=='deg'
    yaw = yaw * 180/pi;
end
return;