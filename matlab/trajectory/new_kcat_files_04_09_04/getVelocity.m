function [Vned, Vh, Vv] = getVelocity(V,Cbn,pitch)
% This function computes the NED velocities and the
% horizontal and vertical components of total
% velocity.
% Change 3/09/04 per T. De Vries suggestion:
Vned = Cbn'*[V;0;0];
Vh   = V*cos(pitch);
Vv   = -Vned(3,1);
return;
