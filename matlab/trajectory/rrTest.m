%rrTest.m:  Relative Rate Test Driver
d2r = pi/180;
Vmag = 7292115e-11*6378137;
roll = 0;
pitch = 0;
hdg = 90*d2r;

Cbn = euler2DCM(roll, pitch, hdg);
lat = 0*pi/180;
V = Cbn*[Vmag;0;0];
sinlat_sq = sin(lat)^2;
coslat_sq = cos(lat)^2;

seg_type = 0;     % Great Circle


for i=1:1000;
alt = (i-1)*6378137/1000;

[rho_n(i),rho_e(i),rho_d(i),alpha_dot(i), Rlat, Rlon] = compRelRates(Vmag, Cbn, lat, alt, seg_type);
% Alternative computation:
[Ra, Rb, f, ecc_sq, Omega] = getWGS84;

rho_n2(i) =  V(2)/Ra*(1-alt/Ra - f*sinlat_sq);  % Works for alt small compared to Ra!!!
rho_e2(i) = -V(1)/Ra*(1-alt/Ra - f*(1-3*coslat_sq));
end

plot(1:6378137/1000:6378137,rho_n,1:6378137/1000:6378137,rho_n2);
legend('De Vries Formula','Savage Approximation');
xlabel('altitude (m)');
ylabel('North Transport Rate (rad/sec)');
title('Deviation of Savage Approx of North Transport Rate from Exact Form at Equator, Heading=90 deg, as function of altitude');
