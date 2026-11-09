A = [0,                     0,  1/R,    0,          0,  0,  0,  Vn/R^2,         0;
     Ve/R*seclat*tanlat,    0,  0,  seclat/R,   0,  0,  0,  Ve*seclat/R^2,  0;
     -Ve*(Ve/R*seclat^2+2*omega_s*coslat),  0,  Vv/R, 2*omega_3,    0,  Av, -Ae,    -(Ve/R)^2*tanlat + Vv*Vn/R^2,   Vn/R;
    Vn*(Ve/R*seclat^2+2*omega_s*coslat)-2*Vv*omega_s*sinlat,    0,  -omega_3+omega_s*sinlat,    Vn/R*tanlat+Vv/R,   -Av,    0,  An, Vn*Ve/R^2*tanlat+Vv*Ve/R^2, omega_1+omega_s*coslat;
    omega_s*sinlat, 0,  0,  -1/R,   0,  omega_3,    -omega_2,   -Ve/R^2,    0;
    0,  0,  1/R,    0,  -omega_3,   0,omega_1,  Vn/R^2, 0;
    (Ve/R)*seclat^2+omega_s*coslat, 0,  0,  tanlat/R,   omega_2,    -omega_1,   0,  (Ve/R^2)*tanlat,    0;
    0,0,0,0,0,0,0,0,1;
    2*omega_s*Ve*sinlat,    0,  -2*Vn/R,    -2*omega_1, Ae, -An,    0,  -(Vn^2+Ve^2)/R^2 + 2*g/R,   0];


