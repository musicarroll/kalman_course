function [Aned] = getInitSensedAccel(alt)
% Computes initial sensed acceleration
% in NED coordinates.  Calls
% getInitGravity.
Aned = zeros(3,1);
Aned(3) = -getInitGravity(alt);

return;
