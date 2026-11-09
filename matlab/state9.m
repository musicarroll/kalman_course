% state9.m Sets up 9-state strapdown INS dynamics in environment for SIMLTIC
% Run simltic.mdl after running this.
% Number of rows in H:
m = 3;

% Earth radius in WGS-84 coordinates:
%
Ra = 6378137.0;
Rb = 6356752.3142;
Re = Ra;
flattening = (Ra-Rb)/Ra;
esq = 2*flattening - flattening^2;
% Acceleration due to gravity:
g = 9.80665;

% Sidereal rate (rad/sec):
omega_s = 1.0027379*2*3.14159/(24*3600);

n=9;
proc_noise = 1;
meas_noise = 1;

I =  eye(n);
P0 = proc_noise * eye(n);
Q =  proc_noise * eye(n);
R =  meas_noise * eye(m);

% Observation Geometry (Measurment) Matrix:
% This creates an upper diagonal matrix of 1's.
for i=1:m
    for j=1:n
        if i<=j
            H(i,j) = 1;
        else
            H(i,j) = 0;
        end
    end
end

% State Dynamics Matrix:

vel_north = input('Enter north velocity(m/sec): ');
vel_east  = input('Enter east velocity(m/sec): ');
lat = input('Enter latitude (deg): ');
lat = lat*pi/180;
sinlat = sin(lat);
coslat = cos(lat);
seclat = sec(lat);
tanlat = tan(lat);

alt = input('Enter altitude (meters): ');

rho_x = trate_east(vel_north,lat,alt);
rho_y = trate_north(vel_east,lat,alt);
rho_z = 0; % free azimuth

% Assume 0 accleration for now:
a_x = 0;
a_y = 0;
a_z = 0;


%alpha = input('Enter the wander angle (deg): ');
%alpha = alpha*pi/180;
alpha  = 0;

omega_x = omega_s*coslat*sin(alpha);
omega_y = omega_s*coslat*cos(alpha);
omega_z = omega_s*coslat;

% Coriolis rates:

cr_x    = 2*omega_x + rho_x;
cr_y    = 2*omega_y + rho_y;
cr_z    = 2*omega_z + rho_z;

% Spatial rates:

sr_x    = omega_x + rho_x;
sr_y    = omega_y + rho_y;
sr_z    = omega_z + rho_z;


% Now build F matrix:

F = [ 0     rho_z  -rho_y     1     0     0     0     0     0;
     -rho_z     0   rho_x     0     1     0     0     0     0;
     rho_x -rho_y       0     0     0     1     0     0     0;
     -g/Re      0       0     0  cr_z -cr_y     0  -a_z   a_y;
         0  -g/Re       0 -cr_z     0  cr_x   a_z     0  -a_x;
       0        0  2*g/Re  cr_y -cr_x     0  -a_y   a_x     0;
       0        0       0     0     0     0     0 -sr_z  sr_y;
       0        0       0     0     0     0  sr_z     0 -sr_x;
       0        0       0     0     0     0  -sr_y  sr_x    0;
   ];
   