function [P1] = covextrap(P0,Phi,Q)
% Error Covariance Extrapolation
P1=Phi*P0*Phi'+Q;
return;