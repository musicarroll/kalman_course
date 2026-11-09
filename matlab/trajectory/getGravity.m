function [g] = getGravity(alt)
% Computes gravity at altitude
% given by alt.
% See "Cruise System Trajectory Generator.doc section
% 5.4.
g1 = 9.780326772; % m/s^2
[Ra, Rb, f, ecc_sq, Omega] = getWGS84; % Ra in meters

g = g1 * Ra^2 / (Ra+alt)^2;

return;
