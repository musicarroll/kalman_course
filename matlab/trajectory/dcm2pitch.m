function [pitch] = dcm2pitch(C,conv);
% dcm2pitch.m is a function that converts the body-to-nav (local level) direction
% cosine matrix to pitch.
pitch = atan2(-C(1,3),sqrt(1-C(1,3)^2));
if conv=='deg'
    pitch = pitch * 180/pi;
end
return;