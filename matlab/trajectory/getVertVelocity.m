function [Vv] = getVertVelocity(V,Cbn)
% This function computes the NED velocities and the
% vertical components of total
% velocity.
% Change 3/09/04 per T. De Vries suggestion:
Vned = Cbn*[V;0;0];
Vv   = -Vned(3,1);
return;
