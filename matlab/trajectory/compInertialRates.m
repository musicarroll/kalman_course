function [omega, omegaC] = compInertialRates(lat, rho)
% Change 12/26/03: Renamed omegaB to omega to reflect the fact
% this quantity is 

Omega_earth = 7.292115e-5; % rad/sec
Omega = [Omega_earth*cos(lat); 0; -Omega_earth*sin(lat)];

omega = Omega + rho;

omegaC = Omega + omega;

return;

