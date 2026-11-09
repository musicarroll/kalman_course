 %    DESCRIPTION:
 %		An m-file script to initialize the workspace with some key
 %		values, but especially it provides the state dynamics A matrix
 %      for inertial error propagation.
 %

% local definitions
%	none	<or definitions if there are any>

% default settings
%	none	<or settings>

% external functions:  none

% external string:     none

% internal functions	none

% global data:  All data is global in the sense that they are placed
%               in the workspace.

% local data:   N/A

% public functions:  N/A

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Function description:  N/A (is a script, not a function)
%	
%
%	Returns:  N/A
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/

% private functions:  N/A

% Building up the A matrix.

% Conversion factors:
rad2deg = 180/3.14159;
rad2mrad = 10e6;

%Input vector:
u=[1,0,0,0,0,0,0,0,0]';

% State Space model.
% Matrix multiplying the control input vector.
B=zeros(9,9);
% Measurement sensitivity matrix, i.e., matrix
% mapping state into output vector.
C=eye(9);
% Matrix mapping control vector into measurement 
% vector:
D=zeros(9,9);

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

sinlat = sin(lat);
coslat = cos(lat);
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

A = [0,                     0,  1/R,    0,          0,  0,  0,  Vn/R^2,         0;
     Ve/R*seclat*tanlat,    0,  0,  seclat/R,   0,  0,  0,  Ve*seclat/R^2,  0;
     -Ve*(Ve/R*seclat^2+2*omega_s*coslat),  0,  Vv/R, 2*omega_3,    0,  Av, -Ae,    -(Ve/R)^2*tanlat + Vv*Vn/R^2,   Vn/R;
    Vn*(Ve/R*seclat^2+2*omega_s*coslat)-2*Vv*omega_s*sinlat,    0,  -omega_3+omega_s*sinlat,    Vn/R*tanlat+Vv/R,   -Av,    0,  An, Vn*Ve/R^2*tanlat+Vv*Ve/R^2, omega_1+omega_s*coslat;
    omega_s*sinlat, 0,  0,  -1/R,   0,  omega_3,    -omega_2,   -Ve/R^2,    0;
    0,  0,  1/R,    0,  -omega_3,   0,omega_1,  Vn/R^2, 0;
    (Ve/R)*seclat^2+omega_s*coslat, 0,  0,  tanlat/R,   omega_2,    -omega_1,   0,  (Ve/R^2)*tanlat,    0;
    0,0,0,0,0,0,0,0,1;
    2*omega_s*Ve*sinlat,    0,  -2*Vn/R,    -2*omega_1, Ae, -An,    0,  -(Vn^2+Ve^2)/R^2 + 2*g/R,   0];

x0=[lat,lon,Vn,Ve,0,0,0,-R,0]';

% testing history update:
%
%
%    $History: Aint.m $
% 
%
