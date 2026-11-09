function [lat, lon] = dcm2latlon(D)
% This function computes the latitude and longitude of
% a point on the earth from the direction cosine matrix
% that transforms the local level coordinates to the earth
% frame.
lat = [atan2(D(2,3),sqrt(D(2,1)^2+D(2,2)^2))];
lon = [atan2(D(1,3),D(3,3))];
return;