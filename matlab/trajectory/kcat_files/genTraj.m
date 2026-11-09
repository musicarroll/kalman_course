% Script genTraj.m
% This Matlab script generates the trajectory
% from the user-specified trajectory file
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

% Change Log:
% 06/12/04:  Replaced Segment and SegIndex as input arguments to several routines with time_step.
% 06/14/04:  Replaced the remaining uses (with one exception) of Segment and SegIndex with appropriate parameters.  This renders the routines more modular and capable of being unit tested more effectively.

if ~exist('file_loaded')
    filename = input('Enter name of trajectory spec file: ','s');
end
load(filename);


% The following calls would be encapsulated in initTraj.m

[roll,pitch,yaw,alpha,Cbn] = getInitAttitude(roll0,pitch0,hdg0);
hdg = yaw;%

% Velocity vector:
[Vned, Vh, Vv] = getInitVelocity(vel0,Cbn,pitch);
V = norm(Vned);

[POSgeo] = getInitPosition(lat0,lon0,alt0);

lat = POSgeo(1);
lon = POSgeo(2);
alt = POSgeo(3);

% Calculate ECEF position:

[POSecef] = geo2ecef(lat,lon,alt);

[g] = getInitGravity(alt);

[Aned] = getInitSensedAccel(alt);

% Change 12/26/03: omegaB (inertial) -> omegaBbar (body) rates:
[omegaBbar] = getInitBodyRates;

[T] = getInitTime;

[SegIndex] = getInitSegIndex;


% Call execTraj.m

Trajectory = zeros(1,20);
RelRates = zeros(1,5);
Cbn_array(1).Cbn = zeros(3,3);
% Change 5/11/04 to be consistent with T. De Vries suggestion:
Cbn_array(1).Cbn = Cbn;

% 1/16/04.  Initialize body rates and accelerations for use with straperr_drv.m:
body_rate_x  = zeros(1,1);
body_rate_y  = zeros(1,1);
body_rate_z  = zeros(1,1);
body_accel_x = zeros(1,1);
body_accel_y = zeros(1,1);
body_accel_z = zeros(1,1);

% Change 12/26/03: Using omegaBbar instead of omegaB, since the latter quantity
% (which has now been renamed to omega) is inertial, not body, while we need
% to record body rates here in the Trajectory array.
% Trajectory(1,:) = [T,Aned(1),Aned(2),Aned(3),Vned(1),Vned(2),Vned(3),...
%        POSecef(1),POSecef(2),POSecef(3),lat, lon, alt,...
%        omegaBbar(1),omegaBbar(2),omegaBbar(3), roll, pitch, hdg];
% 1/16/04
body_rate_x(1) = omegaBbar(1);
body_rate_y(1) = omegaBbar(2);
body_rate_z(1) = omegaBbar(3);
% Change 5/11/04 per T. DeVries suggestion:
accel_vec_body = Cbn*Aned;
body_accel_x(1) = accel_vec_body(1);
body_accel_y(1) = accel_vec_body(2);
body_accel_z(1) = accel_vec_body(3);

i=0;  % Trajectory record index
status = 0; % Determines if simulation is finished.

%%%%% Main Loop %%%%%

while (status==0)

    time_step  = Segment(SegIndex,2);

% Time Logic (6.1):

    if status==1
        break;
    else
        i = i+1;  % increment Trajectory record index.
    end

% Compute pitch angle (6.6):

    [pitch] = compPitch(V, Cbn);

% Compute relative rates (6.7):

    [rho_n,rho_e,rho_d,alpha_dot, Rlat, Rlon] = compRelRates(V, Cbn, lat, alt, seg_type);

    rho = [rho_n;rho_e;rho_d];

% Extrapolate Nav to Body Transformation
% Change 03/26/04: Returning wander angle.  Otherwise, it's value won't
% change upon iteration:

    [Cbn, Cbnl, alpha, true_hdg] = propNav2Body(Cbn, alpha, alpha_dot, hdg, pitch, roll, time_step);

% Velocity Calculations (6.9)

    [Vned, avg_vel, avg_vel_rate] = compVelocity(Vned, V, Cbn, time_step);

% Position Calculations (6.10)

    [POSecef, POSgeo] = compPosition(POSgeo, avg_vel, Rlat, Rlon, time_step);
    lat = POSgeo(1);
    lon = POSgeo(2);
    alt = POSgeo(3);

% Inertial Angular Rates (6.11)

% Change 12/26/03: omegaB renamed to omega for inertial rates:
    [omega, omegaC] = compInertialRates(lat, rho);

% Average Sensed Accelerations (6.12)
% Added 1/2/04: Return acceleration vector in Nav coordinates:

    [accel_vec_body,accel_vec_nav] = compAvgSensedAccel(lat,alt, avg_vel, avg_vel_rate, omegaC, Cbn, Cbnl);
% accel_vec_nav is actually in NED already, since wander angle has been taken out:
    Aned = accel_vec_nav;


% Average Body Rates (6.13)

% Change 12/-26/03: Renamed omegaB to omega to reflect the fact that
% this quantity is really an inertial rate, not a body rate:

    [omegaBbar]=compAvgBodyRates(Cbn,Cbnl, omega, time_step);

% Add record to trajectory:
% Watch out for empty arrays, e.g., roll=[].  This will change
% the number of columns in the record below.
% Change 12/26/03: omegaB (inertial rates) -> omegaBbar (body rates)
% Change 03/26/04: Added true heading, true_hdg
    Trajectory(i,:) = [T,Aned(1),Aned(2),Aned(3),Vned(1),Vned(2),Vned(3),...
                    POSecef(1),POSecef(2),POSecef(3),lat, lon, alt,...
                    omegaBbar(1),omegaBbar(2),omegaBbar(3), ...
                    roll, pitch, true_hdg, hdg];

% Items needed for straperr_drv.m:

    body_rate_x(i)  = omegaBbar(1);
    body_rate_y(i)  = omegaBbar(2);
    body_rate_z(i)  = omegaBbar(3);
    body_accel_x(i) = accel_vec_body(1);
    body_accel_y(i) = accel_vec_body(2);
    body_accel_z(i) = accel_vec_body(3);


    RelRates(i,:) = [T,rho_n,rho_e,rho_d,alpha_dot];
    % Change 5/11/04 to be consistent with T. De Vries suggestion:
    Cbn_array(i).Cbn = Cbn;
    latitude(i)  = lat;
    longitude(i) = lon;
    altitude(i)  = alt;
% Heading Change Logic (6.2):
    final_hdg  = Segment(SegIndex,3);
    turn_accel = Segment(SegIndex,4);


    [hdg,cmd_roll] = procHeadingChange(hdg, time_step, final_hdg, turn_accel,hdg_rate_limit,g, Vh);

% Roll Change Logic (6.3):

    [roll] = procRollChange(roll, cmd_roll, time_step, roll_rate_limit, g, Vh);

% Altitude Change Logic (6.4):
    final_alt   = Segment(SegIndex,7);
    alt_rate    = Segment(SegIndex,8);
    Avn         = Segment(SegIndex,9);
    pitch_angle = Segment(SegIndex,10);

    [alt, Av] = procAltitudeChange(alt, time_step, final_alt, alt_rate, Avn, pitch_angle, pitch_rate_limit, g, V, Vv);

% Velocity Change Logic (6.5):
    Vn = Segment(SegIndex,5); % Final total velocity
    Axn = Segment(SegIndex,6); % Longitudinal acceleration

    [V] = procVelocityChange(V, time_step, Vn, Axn, g, Vh);


    [status,T,SegIndex] = incrTime(T,SegIndex,Segment,num_segs);


end

%%%% End Main Loop

body_rate_x = body_rate_x';
body_rate_y = body_rate_y';
body_rate_z = body_rate_z';
body_accel_x = body_accel_x';
body_accel_y = body_accel_y';
body_accel_z = body_accel_z';

disp('Trajectory Generation Complete.');
if exist('author')
    disp(['Scenario author: ',author]);
    disp(['Description: ',description]);
    disp(final_comments);
end


% Call saveTraj.m
