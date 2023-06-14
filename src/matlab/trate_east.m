function [y] = trate_east(vel_north,lat,alt)
% Function to compute the east transport rate, i.e., the rotational
% rate about the east axis due to vehicle velocity.
Ra = 6378137.0;
Rb = 6356752.3142;
Re = Ra;
flattening = (Ra-Rb)/Ra;
esq = 2*flattening - flattening^2;
e  = 1/298;
coslat = cos(lat);
y = -vel_north/Re * (1-alt/Re - e*(1-3*coslat^2));
return;