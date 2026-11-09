function [roll] = dcm2roll(C,conv);
% dcm2roll.m is a function that converts the body-to-nav (local level) direction
% cosine matrix to roll.
roll = atan2(C(2,3),C(3,3));
if conv=='deg'
    roll = roll * 180/pi;
end
return;