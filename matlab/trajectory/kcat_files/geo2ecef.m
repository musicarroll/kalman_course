function [POSecef] = geo2ecef(lat,lon,alt)

[Rlat, Rlon, ae, D] = getRadii(lat,alt);  % calculate new radii
    
POSecef = zeros(3,1);
POSecef(1) = (ae/D + alt)*cos(lat)*cos(lon);
POSecef(2) = (ae/D+alt)*cos(lat)*sin(lon);
POSecef(3) = Rlat*sin(lat);

return;
