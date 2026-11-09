% Building up the A matrix.

% Earth radius in WGS-84 coordinates:
%
Ra = 6378137.0;
Rb = 6356752.3142;
Re = Ra;
flattening = (Ra-Rb)/Ra;
esq = 2*flattening - flattening^2;

% height nom = 1000 ft
% converted to meters:
h=1000*.3048;
R = Re + h;

% latitude:
lat = 34.00*3.14159/180;

% longitude:
lon = -120.0*3.14159/180;

% East velocity @ 100 fps:
Ve = 100*.3048;

% North velocity @ 100 fps:
Vn = 100*.3048;

% Vertical velocity:  0 for level flight.
Vv = 0;

% North component of specific force:
An = 0;

% East component of specific force:
Ae = 0;

% Vertical component of specific force:
Av = 0;

% Acceleration due to gravity:
g = 9.80665;

slat = sin(lat);
clat = cos(lat);
seclat = sec(lat);
tanlat = tan(lat);

Rp = Ra/sqrt(1-esq*sinlat^2);
Rm = Ra*(1-esq)/sqrt(1-esq*sinlat^2);

% Sidereal rate (rad/sec):
omega_s = 1.0027379*2*3.14159/(24*3600);

phi_dot = Ve*seclat/(Rp+h);
omega_1 = omega_s*coslat+Ve/(Rp+h);
omega_2 = -Vn/(Rm+h);
omega_3 = -omega_s*sinlat - Ve*tanlat/(Rp+h);

% Now build the A matrix:


