function [Phi] = maketrans(trate_level,erate_level,accel_level)
% This function generates a 9x9 transition matrix for propagating
% inertial errors.
% Error states:  pos err(3), vel err(3), tilt/az err (3)
% Inputs:  trate_level = transport rate in level axes
%          erate_level = earth rate in level axes
%          accel_level = acceleration in level axes
Omega_EL = skew(trate_level);
Omega_IE = skew(erate_level);
A_L      = skew(accel_level);
R        = 6.378e6/2; % earth radius: spherical earth is OK for error dynamics
U_L      = skew([0 0 1]);
Phi = [-Omega_EL 1/R*U_L zeros(3,3);
        zeros(3,3) -(2*Omega_IE+Omega_EL) A_L;
        Omega_IE 1/R*U_L -(Omega_IE + Omega_EL)];
return;


