function [F_array] = build17StateSysDyn(Trajectory,RelRates)
%     Trajectory(i,:) = [T,Aned(1),Aned(2),Aned(3),Vned(1),Vned(2),Vned(3),...
%                    POSecef(1),POSecef(2),POSecef(3),lat, lon, alt,...
%                    omegaBbar(1),omegaBbar(2),omegaBbar(3), ...
%                    roll, pitch, true_hdg, hdg];

% Earth radius in WGS-84 coordinates:
%
[m,n]=size(Trajectory);
zero_col = zeros(m,1);
one_col  = ones(m,1);

Ra = 6378137.0;
Rb = 6356752.3142;
Re = Ra;
% Sidereal rate (rad/sec):
Omega = 1.0027379*2*3.14159/(24*3600); %=7.292109627918981e-05
Omega_sq = Omega^2;

for i=1:m
    alt = Trajectory(i,13);
    g = getInitGravity(alt);
    omega_s = sqrt(g./Re);
    omega_s_sq = omega_s.^2;

    lat = Trajectory(i,11);
    lon = Trajectory(i,12);

    An  = Trajectory(i,2);
    Ae  = Trajectory(i,3);
    Ad  = Trajectory(i,4);

    Vn  = Trajectory(i,5);
    Ve  = Trajectory(i,6);
    Vd  = Trajectory(i,7);
    
    % Cnvert to Free Azimuth Coordinates:
    phdg = Trajectory(i,20);
    thdg = Trajectory(i,19);
    alpha = phdg - thdg;
    ca = cos(alpha);
    sa = sin(alpha);
    
    Rot = [ca -sa; sa ca];
    
    V = Rot'*[Vn; Ve];
    A = Rot'*[An; Ae];
    Vx = V(1);
    Vy = V(2);
    Vz = Vd;
    
    Ax = A(1);
    Ay = A(2);
    Az = Ad;
    
    coslat = cos(lat);
    sinlat = sin(lat);

    Omega_n = Omega * coslat;
    Omega_e = 0;
    Omega_d = -Omega * sinlat;

    rho_n   =  RelRates(i,2);
    rho_e   =  RelRates(i,3);
    rho_d   =  RelRates(i,4);

    omega_n = Omega_n + rho_n;
    omega_e = Omega_e + rho_e;
    omega_d = Omega_d + rho_d;

    omega_sq = omega_n.^2 + omega_e.^2 + omega_d.^2;

    [Rlat, Rlon, ae, D] = getRadii(lat,alt);

% Check sign consistency!!!
    rho_dot_n = Ae./Rlon;
    rho_dot_e = -An./Rlat;
    rho_dot_d = 0;

% Now build 17x17 intertial error state
% dynamics matrix block by block:

    Z = zeros(3); 
    I = eye(3);

    F21(1,1) = An./Re-omega_s_sq+omega_sq - Omega.^2 ...
            - (omega_n + Omega_n).*rho_n;
    F21(1,2) = rho_dot_d - (omega_e + Omega_e).*rho_n;
    F21(1,3) = -rho_dot_e - (omega_d + Omega_d).*rho_n;
    F21(2,1) = -rho_dot_d - (omega_n + Omega_n).*rho_e;
    F21(2,2) = An./Re-omega_s_sq+omega_sq - Omega.^2 ...
            - (omega_e + Omega_e).*rho_e;
    F21(2,3) = rho_dot_n - (omega_d + Omega_d).*rho_e;
    F21(3,1) = rho_dot_e - An./Re - (omega_n + Omega_n).*rho_d;
    F21(3,2) = -rho_dot_n - Ae./Re - (omega_e + Omega_e).*rho_d;
    F21(3,3) = omega_sq - Omega_sq + 2.*omega_s_sq ...
            - (omega_d + Omega_d).*rho_d;
    F22 = [0 2.*omega_d -2.*omega_e; ...
          -2.*omega_d 0 2.*omega_n; ...
           2.*omega_e -2.*omega_n 0 ];

    F23 = [ 0 -Ad Ae; Ad 0 -An; -Ae An 0];

    F31 = [ -omega_d./Re 0 0; 0 -omega_e./Re 0; omega_n./Re omega_e./Re 0];

    F32 = [ 0 1./Re 0; 1./Re 0 0; 0 0 0];

    F33 = [ 0 omega_d -omega_e; -omega_d 0 omega_n; omega_e -omega_n 0];

% Put the blocks together:

    B2POS = [ 0 0 -V

    F_array(:,:,i) = [   Z   I   Z   Z; ...
      F21 F22 F23; ...
      F31 F32 F22 ];
end

return;

