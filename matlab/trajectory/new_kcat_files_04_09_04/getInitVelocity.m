function [Vned, Vh, Vv] = getInitVelocity(vel0,Cbn,pitch)
% This function computes the NED velocities and the
% horizontal and vertical components of total
% velocity.
vel  = vel0;
% Change 3/9/04 per T. D. Vries suggestion:
Vned = Cbn'*[vel;0;0];
Vh   = vel*cos(pitch);
Vv   = -Vned(3,1);
return;
