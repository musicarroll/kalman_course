function [pbg] = plumbbobg(alt,sinlat)
% This function computes the plumb-bob gravity 
% for use in straperr.mdl.

Re = 6.378e6;
Rp = 6.357e6;
e  = sqrt(1-Rp^2/Re^2);
R = Re*(1-e*sinlat^2)+alt;
G1 = 9.80665;
u = G1*Re^2;
We = 7.2921151467e-5;	
J2 = 0.00108;
G2 = (3/2)*J2*G1;
G3 = Re*We^2;

%pbg = G1*(1 - 2*alt/Re + 2*e*sinlat^2) + ...
%            G2*(1-3*sinlat^2) ...
%                - G3*(1-e*sinlat^2+alt/Re)*(1-sinlat^2);
coslat = sqrt(1-sinlat^2);
cos2lat = coslat^2 - sinlat^2;
pbg = (u/R^2)*(1-(3/4)*J2*(1-3*cos2lat))-R*We^2*coslat^2;

return;