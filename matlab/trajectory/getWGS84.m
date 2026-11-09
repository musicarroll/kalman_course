function [Ra, Rb, f, ecc_sq, Omega] = getWGS84

% Earth radius in WGS-84 coordinates:
% Change Log:
% 06/18/04:  Changed Omega to match WGS 84 spec more closely.

Ra = 6378137.0; % semimajor axis
Rb = 6356752.3142; % smiminor axis
Re = Ra;
f = (Ra-Rb)/Ra; % flattening
ecc_sq = (2*f - f^2);
% Second eccentricity:
% e_prime_sq = (2*f - f^2)/(1-f)^2;
%Omega = 1.0027379*2*pi/(24*3600); %=7.292109627918981e-05
% The following is more consistent with WGS84 spec:
Omega = 7292115e-11;
return;



