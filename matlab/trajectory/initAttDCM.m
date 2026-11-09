function [C] = initAttDCM(roll,pitch,platform_hdg,conv)
% function [C] =initAttDCM(roll,pitch,platform_hdg) generates 
% body to local level direction cosine matrix from inputs of
% roll, pitch and platform_hdg relative to the local level 
% axes.
phi   = roll;
theta = pitch;
psi_p = platform_hdg;

if strcmp(conv,'deg') % arguments are in degrees.
    phi   = phi * pi/180;
    theta = theta * pi/180;
    psi_p = psi_p * pi/180;
end
% Set up the sines and cosines:
ctheta = cos(theta);
cphi   = cos(phi);
cpsi   = cos(psi_p);
stheta = sin(theta);
sphi   = sin(phi);
spsi   = sin(psi_p);

% Fill the DCM:

C(1,1) = ctheta*cpsi;
C(1,2) = -cphi*spsi+sphi*stheta*cpsi;
C(1,3) = sphi*spsi + cphi*stheta*cpsi;
C(2,1) = ctheta * spsi;
C(2,2) = cphi*cpsi + sphi*stheta*spsi;
C(2,3) = -sphi*cpsi+cphi*stheta*spsi;
C(3,1) = -stheta;
C(3,2) = sphi*ctheta;
C(3,3) = cphi*ctheta;

return;