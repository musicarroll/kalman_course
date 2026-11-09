function [Rlat, Rlon, ae, D] = getRadii(lat,alt)

F = 6.69437999e-3;
ae = 6378137; % meters
D = sqrt(1-F.*sin(lat).^2);

Rlat = ae.*(1-F)./D + alt;
Rlon = ae./D.^3 + alt;

return;
