function [Phi_array] = build28StateTransMat(Trajectory,RelRates)
% The fields of the Trajectory array are:
%Number     Symbol	Data				Units		Notes
%    1	        	Time				seconds
%    2	        	North Sensed Acceleration	meters/sec^2
%    3	        	East Sensed Acceleration	meters/sec^2
%    4	        	Down Sensed Acceleration	meters/sec^2
%    5	        	North Velocity		meters/sec
%    6	        	East Velocity		meters/sec
%    7	         	Down Velocity		meters/sec
%    8	        	X ECEF Position		meters
%    9	        	Y ECEF Position		meters
%   10	        	Z ECEF Position		meters
%   11	        	Latitude			radians
%   12	         	Longitude			radians
%   13	         	Altitude			meters
%   14	         	X Body Rate			rad/sec
%   15	         	Y Body Rate			rad/sec
%   16	         	Z Body Rate			rad/sec
%   17	          	Roll				radians
%   18	          	Pitch				radians
%   19	          	True Heading		radians
%   20.             Platform Heading    radians
%     Trajectory(i,:) = [T,Aned(1),Aned(2),Aned(3),Vned(1),Vned(2),Vned(3),...
%                    POSecef(1),POSecef(2),POSecef(3),lat, lon, alt,...
%                    omegaBbar(1),omegaBbar(2),omegaBbar(3), ...
%                    roll, pitch, true_hdg, hdg];

% Maintenance Notes:
% 1)  All multiplications, divisions and exponentiations are vectorized,
% although this is currently not necessary.
% 2) This implementation follows that outlined for the 28-state Kalman
% filter described in the paper "The CMIGITS INS/GPS Kalman Filter" by
% Michael L. Carroll and Dr. Tom De Vries.
% 3) The state transition matrix Phi for time-invariant
% systems equals the matrix exponential of F*Dt, where
% F is the state dynamics matrix (continuous) and Dt is the
% time increment.  However, since our dynamics are time
% varying (see Markov bias states and rate coupling terms),
% we have altered KCAT to become KCAT-TV and expect an
% explicitly provided Phi instead of F.

% Earth radius in WGS-84 coordinates:
%
[m,n]=size(Trajectory);
zero_col = zeros(m,1);
one_col  = ones(m,1);

% Get WGS Ellipsoid parameters:
[Ra, Rb, f, ecc_sq, Omega] = getWGS84;
Re = Ra;
Omega_sq = Omega^2;

Dt = 1;  % Note hard-coded Kalman extrapolation delta time
% Gyro Scale Factor and Misalignment matrix:
V=[.001,.001,.001];
Kg = diag(V,0);  % Note hard-coding!!!

for i=1:m
    alt        = Trajectory(i,13);
    g          = getGravity(alt);
    omega_s    = sqrt(g./Re);
    omega_s_sq = omega_s.^2;

    lat = Trajectory(i,11);
    lon = Trajectory(i,12);

    An  = Trajectory(i,2);
    Ae  = Trajectory(i,3);
    Ad  = Trajectory(i,4);

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
    Vn  = Trajectory(i,5);
    Ve  = Trajectory(i,6);
    Vd  = Trajectory(i,7);

    % Convert to Free Azimuth Coordinates:
    phdg = Trajectory(i,20);
    thdg = Trajectory(i,19);
    alpha = phdg - thdg;  % Wander angle

    ca = cos(alpha);
    sa = sin(alpha);

    Rot = [ca -sa; sa ca];

    V = Rot'*[Vn; Ve];

    Vx = V(1);
    Vy = V(2);
    Vz = Vd;

    % Accelerations:

    A = Rot'*[An; Ae];
    Ax = A(1);
    Ay = A(2);
    Az = Ad;

    % Relative Rates:
    % For error analysis, spherical earth can be assumed:
    rho_x = -Vy/(Re+alt);
    rho_y = Vx/(Re+alt);
    rho_z = 0; % for wander azimuth implementation

    % Earth rate:
    Omega_x = Omega_n * sa;
    Omega_y = Omega_n * ca;
    Omega_z = Omega * sinlat;

% Now build canoncial 28x28 inertial error state
% dynamics matrix block by block:

% Position Error Block:

    POSERR = [ 1 0 -rho_y Dt 0 0   0 -Vz  Vy ...
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; ... % end 1st row
               0 1  rho_x 0 Dt 0 Vz    0 -Vx ...
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; ... % end 2nd row
           rho_y -rho_x 1 0 0 Dt -Vy Vx 0 ...
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];    % end 3rd row

% Gyro Bieas Drift Block:
% Gyro biases modeled as first order Markov with bias
% correlation time of tau_g.  This is currently hard-coded!
tau_g = 40;  % 40 sec correlation time.
% States 23-25:
    z22 = zeros(1,22);
    z3  = zeros(1,3);
    GYROBIAS = [ z22 exp(-Dt/tau_g)         0            0    z3;...
                 z22   0        exp(-Dt/tau_g)      0         z3;...
                 z22   0               0      exp(-Dt/tau_g)  z3];


% Gyro Scale Factor Block:
    % States 17-19:
    z3x16 = zeros(3,16);
    z3x9  = zeros(3,9);
    GYROSF = [z3x16 eye(3) z3x9];
% Total Gyro Error Vector in Body Coordinates:
    omega_bx = Trajectory(i,14); % body rates
    omega_by = Trajectory(i,15);
    omega_bz = Trajectory(i,16);
    total_gyro_err_body = ...
        [exp(-Dt/tau_g); exp(-Dt/tau_g); exp(-Dt/tau_g)]...
        + Kg*[omega_bx;omega_by;omega_bz];
% Convert to Nav coordinates (local level):
    roll  = Trajectory(i,17);
    pitch = Trajectory(i,18);
    thdg  = Trajectory(i,19);
    [Cbn] = euler2DCM(roll, pitch, thdg);
    total_gyro_err_nav = Cbn*total_gyro_err_body;

% Attitude Error Block (psi_star):
% Convert total spatial rates to nav coordinates:
    omega_nav = Cbn'*[omega_n;omega_e;omega_d];
    % States 6-9:
    ATTERR = [ 0 0 0 0 0 0 1 omega_nav(3)*Dt -omega_nav(2) 0 0 0 0 0 0 0 ...
               Cbn(1,1)*Dt Cbn(1,2)*Dt Cbn(1,3)*Dt ... % 17-19
                 0 0 0 0 0 0 0 0 0;... % end 1st row
               0 0 0 0 0 0 -omega_nav(3)*Dt 1 omega_nav(1)*Dt 0 0 0 0 0 0 0 ...
               Cbn(2,1)*Dt Cbn(2,2)*Dt Cbn(2,3)*Dt ...
                 0 0 0 0 0 0 0 0 0;... % end 2nd row
               0 0 0 0 0 0 omega_nav(2)*Dt -omega_nav(1)*Dt 1 0 0 0 0 0 0 0 ...
               Cbn(3,1)*Dt Cbn(3,2)*Dt Cbn(3,3)*Dt ...
                 0 0 0 0 0 0 0 0 0]; % end 3rd row

% Accelerometer Bias Drift Block:
% States 26-28:
% Acceleromater biases modeled as first order Markov with bias
% correlation time of tau_a.  This is currently hard-coded!
tau_a = 40;  % 40 sec correlation time.
% States 26-28:
    z25 = zeros(1,25);
    ACCELBIAS = [z25 exp(-Dt/tau_a)         0            0;...
                 z25            0        exp(-Dt/tau_a)      0;...
                 z25            0               0      exp(-Dt/tau_a)];


% Accelerometer Scale Factor Block:
% States 20-22
    z3x19 = zeros(3,19);
    z3x6  = zeros(3,6);
    ACCELSF = [ z3x19 eye(3) * Dt z3x6];

% Velocity Error Block:
% This is the most complicated block because of dependencies on the foregoing
% blocks.  See section 7.3 of Cmigits paper.  We use Eq. 7.24
% instead of Eq. 7.12.
% a) Coriolis rates (used in Phi to propagate states 4-6 and 7-9):
    omega_c_x = 2*Omega_x + rho_x;
    omega_c_y = 2*Omega_y + rho_y;
    omega_c_z = 2*Omega_z + rho_z;
% b) Accerlation and gravity dependent terms:
    omega_schuler_sq = g/Ra;
% See attitude and accel bias terms
% c)
    VELERR = [0 0 0 0 omega_c_z -omega_c_y 0 -Omega_z/Dt-g*Dt Omega_y/Dt ...
              0 0 0 0 0 0 0 0 0 0 0 0 0 ...
	      1/Dt 0 0 1 0 0; ... % end 1st row
	      0 0 0 -omega_c_z 0 omega_c_x Omega_z/Dt+g*Dt 0 -Omega_x/Dt ...
              0 0 0 0 0 0 0 0 0 0 0 0 0 ...
	      0 1/Dt 0 0 1 0; ... % end 2st row
	      0 0 2*omega_schuler_sq*Dt omega_c_y -omega_c_x 0 -Omega_y/Dt Omega_x/Dt 0 ...
              0 0 0 0 0 0 0 0 0 0 0 0 0 ...
	      0 0 1/Dt 0 0 1]; % end 3rd row

% GPS Error Block (simply use the identity matrix for now; see REFSTATES):


% Misc. Reference States: 7 states 10-16:
    REFSTATES = [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ...
    		 0 0 0 0 0 0 0 0 0 0 0 0; % end 1st row;
                 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ...
    		 0 0 0 0 0 0 0 0 0 0 0 0; % end 2st row;
                 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 ...
    		 0 0 0 0 0 0 0 0 0 0 0 0; % end 3rd row;
                 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 ...
    		 0 0 0 0 0 0 0 0 0 0 0 0; % end 4th row;
                 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ...
    		 0 0 0 0 0 0 0 0 0 0 0 0; % end 5th row;
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ...
    		 0 0 0 0 0 0 0 0 0 0 0 0; % end 6th row;
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ...
    		 0 0 0 0 0 0 0 0 0 0 0 0]; % end 7th row;

% GPS as exponentially-correlated terms (States 11-13):
tau_gps_freq_rate_err = 10;
REFSTATES(2,11) = exp(-Dt/tau_gps_freq_rate_err);

tau_gps_clock_freq_err = 10;
REFSTATES(3,12) = exp(-Dt/tau_gps_clock_freq_err);

tau_gps_clock_bias_err = 10;
REFSTATES(4,13) = exp(-Dt/tau_gps_clock_bias_err);

% Put the blocks together:


    Phi_array(:,:,i) = [   POSERR; ...    % 3x28
                           VELERR; ...    % 3x28
                           ATTERR; ...    % 3x28
			   REFSTATES; ... % 7x28
			   GYROSF; ...    % 3x28
			   ACCELSF; ...   % 3x28
			   GYROBIAS; ...  % 3x28
			   ACCELBIAS ];   % 3x28
end

return;

