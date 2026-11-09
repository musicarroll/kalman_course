function [D0] = initPosDCM(lat, lon, wander,conv)
% initPosDCM.m produces an initial direction cosine matrix corresponding to 
% the operator specified initial latitutde, longitude and wander angle.
% You need to run this before running the straperr.mdl Simulink model.
if strcmp(conv,'deg') % arguments are in degrees.
    lat = lat * pi/180;
    lon = lon * pi/180;
    wander = wander * pi/180;
end
D0 = eye(3); 
D0(1,1) = cos(lon)*cos(wander) - sin(lon)*sin(lat)*sin(wander);
D0(1,2) = -cos(lon)*sin(wander) - sin(lon)*sin(lat)*cos(wander);
D0(1,3) = sin(lon)*cos(lat);
D0(2,1) = cos(lat)*sin(wander);
D0(2,2) = cos(lat)*cos(wander);
D0(2,3) = sin(lat);
% Third row is cross product of first two rows:
D0(3,:) = cross(D0(1,:),D0(2,:));
return;
