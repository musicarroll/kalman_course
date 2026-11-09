function [lon] = dcm2lon(D,conv)
% This function computes the latitude and longitude of
% a point on the earth from the direction cosine matrix
% that transforms the local level coordinates to the earth
% frame.
lon = atan2(D(1,3),D(3,3));
if conv=='deg'
    lon = lon*180/pi;
end
return;