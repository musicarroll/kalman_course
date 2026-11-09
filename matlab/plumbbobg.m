function [pbg] = plumbbobg(alt,lat)
% This function computes the plumb-bob gravity
%
sinlat = sin(lat); % lat must be in radians
coslat = cos(lat);
cos2lat = coslat.^2 - sinlat.^2;

% WGS-84

Re = 6.378e6; % Earth equatorial radius
Rp = 6.357e6; % Earth polar radius

e  = sqrt(1-Rp.^2/Re.^2); % Eccentricity
R  = Re.*(1-e.*sinlat.^2)+alt; % Radial distance from Earth's center to vehicle
G1 = 9.80665; % Mean gravity at Earth's surface on equator
u  = G1.*Re.^2;
We = 7.2921151467e-5; % Earth's rotational rate in rad/sec
J2 = 0.00108; % Empirical constant
G2 = (3/2).*J2.*G1;
G3 = Re.*We.^2;

% pbg = G1*(1 - 2*alt/Re + 2*e*sinlat^2) + ...
%            G2*(1-3*sinlat^2) ...
%                - G3*(1-e*sinlat^2+alt/Re)*(1-sinlat^2);
pbg = (u./R.^2).*(1-(3/4).*J2.*(1-3.*cos2lat))-R.*We.^2.*coslat.^2;

return;
