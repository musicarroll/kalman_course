function [phiout] = phiharmosc(zeta,omega,T)

F=[0 1;-omega^2 -2*zeta*omega];
phiout = expm(F*T);
return;
