% plottraj.m
% A script to plot KCAT trajectory generator output.
% Assumes workspace is loaded with trajectory data.
% Uses Trajectory array.
% Interpretation of Trajectory:
%Number     Symbol	Data				Units		Notes	
%    1	        	Time				seconds
%    2	        	North Sensed Acceleration	meters/sec2	   1
%    3	        	East Sensed Acceleration	meters/sec2	   1
%    4	        	Down Sensed Acceleration	meters/sec2	   1
%    5	        	North Velocity		meters/sec
%    6	        	East Velocity			meters/sec
%    7	         	Down Velocity		meters/sec
%    8	        	X ECEF Position		meters
%    9	        	Y ECEF Position		meters
%   10	        	Z ECEF Position		meters
%   11	        	Latitude			radians
%   12	         	Longitude			radians		    2	
%   13	         	Altitude			meters
%   14	         	X Body Rate			rad/sec		    1
%   15	         	Y Body Rate			rad/sec		    1
%   16	         	Z Body Rate			rad/sec		    1   	
%   17	          	Roll				radians		    2
%   18	          	Pitch				radians
%   19	          	True Heading		radians		    2
%   20.                 Platform Heading
backup_charts=1;

% Plot Acceleration
figure;
subplot(3,1,1), plot(Trajectory(:,1),Trajectory(:,2));
legend('North');
ylabel('m/sec^2');
title('Sensed Acceleration: North-East-Down');
subplot(3,1,2), plot(Trajectory(:,1),Trajectory(:,3));
legend('East');
ylabel('m/sec^2');
subplot(3,1,3), plot(Trajectory(:,1),Trajectory(:,4));
legend('Down');
ylabel('m/sec^2');
xlabel('sec');

% Plot Velocity
figure;
subplot(3,1,1), plot(Trajectory(:,1),Trajectory(:,5));
legend('North');
ylabel('m/sec');
title('Velocity: North-East-Down');
subplot(3,1,2), plot(Trajectory(:,1),Trajectory(:,6));
legend('East');
ylabel('m/sec');
subplot(3,1,3), plot(Trajectory(:,1),Trajectory(:,7));
legend('Down');
ylabel('m/sec');
xlabel('sec');

% Plot Position (Lat/Lon/Alt)
rad2deg = 180/pi;
figure;
subplot(3,1,1), plot(Trajectory(:,1),Trajectory(:,11)*rad2deg);
legend('Latitude');
ylabel('deg');
title('Position: Lat/Lon/Alt');
subplot(3,1,2), plot(Trajectory(:,1),Trajectory(:,12)*rad2deg);
legend('Longitude');
ylabel('deg');
subplot(3,1,3), plot(Trajectory(:,1),Trajectory(:,13));
legend('Altitude');
ylabel('meters');
xlabel('sec');

% Planview Map
figure;
plot(Trajectory(:,12)*rad2deg,Trajectory(:,11)*rad2deg);
xlabel('Longitude (deg)');
ylabel('Latitude (deg)');

% Plot Body Rates: XYZ
figure;
subplot(3,1,1), plot(Trajectory(:,1),Trajectory(:,14)*rad2deg);
legend('X');
ylabel('deg/sec');
title('Body Rates: XYZ');
subplot(3,1,2), plot(Trajectory(:,1),Trajectory(:,15)*rad2deg);
legend('Y');
ylabel('deg/sec');
subplot(3,1,3), plot(Trajectory(:,1),Trajectory(:,16)*rad2deg);
legend('Z');
ylabel('deg/sec');
xlabel('sec');

% Plot Attitude: Roll, Pitch and True Heading
figure;
subplot(3,1,1), plot(Trajectory(:,1),Trajectory(:,17)*rad2deg);
legend('Roll');
ylabel('deg');
title('Attitude: Roll-Pitch-Heading');
subplot(3,1,2), plot(Trajectory(:,1),Trajectory(:,18)*rad2deg);
legend('Pitch');
ylabel('deg');
subplot(3,1,3), plot(Trajectory(:,1),Trajectory(:,19)*rad2deg);
legend('Heading');
ylabel('deg');
xlabel('sec');

% Miscellaneous Diagnostic Plots
if backup_charts
    figure;
    subplot(4,1,1), plot(RelRates(:,1),RelRates(:,2)*rad2deg);
    legend('\rho_N');
    ylabel('deg/sec');
    title('Relative Rates');
    subplot(4,1,2), plot(RelRates(:,1),RelRates(:,3)*rad2deg);
    legend('\rho_E');
    ylabel('deg/sec');
    subplot(4,1,3), plot(RelRates(:,1),RelRates(:,4)*rad2deg);
    legend('\rho_D');
    ylabel('deg/sec');
    subplot(4,1,4), plot(RelRates(:,1),RelRates(:,5)*rad2deg);
    legend('\alpha dot');
    ylabel('deg/sec');
    xlabel('sec');
end
