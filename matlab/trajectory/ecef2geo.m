function [lat, lon, alt] = ecef2geo(x, y, z)
% This m-file function converts ECEF coordinates to Geodeteic lat/lon/alt
% Usage:  [lat, lon, alt] = ecef2geo(x, y, z)
% Returns lat,lon in radians and alt in meters.  x,y,z must be in meters.

len = length(x);

a=6378137.0;
b=6356752.3142;
f=(a-b)/a;
esq=2*f-f^2;
eprimesq=(a^2-b^2)/b^2;

X=x;
Y=y;
Z=z;

p=sqrt(X.^2+Y.^2);
theta=atan2(Z.*a,p.*b);
sintheta=sin(theta);
costheta=cos(theta);

sintheta_cubed = sintheta.*sintheta.*sintheta;
costheta_cubed = costheta.*costheta.*costheta;

phi=atan2(Z+eprimesq.*b.*sintheta_cubed,p-esq.*a.*costheta_cubed);
sinphi=sin(phi);
N=a./sqrt(ones(len,1)-esq*sinphi.^2);

lambda = atan2(Y,X);
h = p ./ cos(phi) - N;

lat=phi;
lon=lambda;
alt=h;
return
