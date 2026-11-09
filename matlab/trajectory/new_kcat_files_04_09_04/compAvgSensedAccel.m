function [accel_vec_body, accel_vec_nav] = compAvgSensedAccel(lat,alt, avg_vel, avg_vel_rate, omegaC, Cbn, Cbnl)
% Section 6.12.
g1 = 9.780326772; % m/s^2
g2 = 1.889413783e-2; % m/s^2
[Rlat, Rlon, ae, D] = getRadii(lat,alt);
g0 = (g1+g2*sin(lat)^2)/D;
% Change on 1/2/04: Found bug: Was (ae+alt^2) in
% denominator.  Should be (ae+alt)^2:
g = g0*ae^2/(ae+alt)^2;

g_vec = [0;0;g];
coriolis_accel = cross(omegaC,avg_vel);
accel_vec_nav = avg_vel_rate + coriolis_accel - g_vec; % navigation sensed accelerations
% Chan ge 5/11/04 per T. De Vries suggestion:
accel_vec_body = ((Cbn+Cbnl)/2)*accel_vec_nav;  % body sensed accelerations

return;
