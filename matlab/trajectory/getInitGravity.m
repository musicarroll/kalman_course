function [g] = getInitGravity(alt)
% Computes gravity at altitude
% given by alt.
% See "Cruise System Trajectory Generator.doc section
% 5.4.
g1 = 9.780326772; % m/s^2
ae = 6378137; % meters

g = g1 .* ae.^2 ./ (ae+alt).^2;

return;
