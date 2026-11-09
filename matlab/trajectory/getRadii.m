function [Rlat, Rlon, ae, D] = getRadii(lat,alt)

[Ra, Rb, f, ecc_sq, Omega] = getWGS84;
F = ecc_sq;
ae = Ra;  % 6378137; % meters semimajor axis
D = sqrt(1-F.*sin(lat).^2);
% There appears to be a mistake in De Vries "Cruise System Trajectory
% Generator:  The D^3 factor needs to be Rlat, not Rlon.
% Was:
%Rlat = ae.*(1-F)./D + alt;
%Rlon = ae./D.^3 + alt;
% Is
Rlat = ae.*(1-F)./D.^3 + alt;
Rlon = ae./D + alt;

return;
