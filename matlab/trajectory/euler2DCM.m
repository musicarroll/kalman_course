function [DCM] = euler2DCM(roll, pitch, yaw)
% Function converts Euler angles in radians to 
% a direction cosine matrix:
cr = cos(roll);
sr = sin(roll);
cp = cos(pitch);
sp = sin(pitch);
cy = cos(yaw);
sy = sin(yaw);
% The following was from the 10/02/03 C-MIGITS Kalman 
% filter document and cannot be correct, since for roll,
% pitch, and yaw values of 0,0,0, the identity matrix 
% should result and does not with the following:
%DCM=[cp*sy  sy*sp*sr-cy*cr  sy*sp*cr-cy*sr;
%     cp*cy  cy*sp*sr-sy*cr  cy*sp*cr+sy*sr;
%     sp     -cp*sr          -cp*cr];

% From Kuiper's "Quaternions and Rotation Sequences"
DCM = zeros(3,3);
DCM(1,:)=[cp*cy,          sy*cp,           -sp];
DCM(2,:)=[cy*sp*sr-sy*cr, sy*sp*sr+cy*cr,   cp*sr];
DCM(3,:)=[cy*sp*cr+sy*sr, sy*sp*cr-cy*sr,   cp*cr];
% Change 5/19/04:  To make other uses of Cbn conform with De Vries usage.
DCM = DCM';
return;
