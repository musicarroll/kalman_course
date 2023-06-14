function [y] = trate_north(vel_east,lat,alt)
% Function to compute the east transport rate, i.e., the rotational
% rate about the east axis due to vehicle velocity.
Ra = 6378137.0;
Rb = 6356752.3142;
Re = Ra;
flattening = (Ra-Rb)/Ra;
esq = 2*flattening - flattening^2;
e  = 1/298;
sinlat = sin(lat);
y = vel_east/Re * (1-alt/Re - e*sinlat^2);
return;