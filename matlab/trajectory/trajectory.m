% Plots Tom De Vries Trajectory Data.
% Revision History:
% Date      Author      Description
% 04/09/02  M. Carroll  Initial Release

% Format of data (SDID1.dat)

% 1    Time, sec
% 2    Attitude Type Indicator, 3 = R,P,H
% 3    GPS Type Indicator, 0 = no GPS data
% 4    North Accel, ft/sec^2
% 5    East Accel
% 6    Down Accel
% 7    North Velocity, ft/sec
% 8    East Velocity
% 9    Down Velocity
%10    Latitude, deg
%11    Longitude, deg
%12    Altitude, ft
%13    X Body Rate, rad/sec
%14    Y Body Rate
%15    Z Body Rate
%16    Roll, deg
%17    Pitch
%18    Heading


[numTimeSteps,numCols] = size(Sdid1);

% Assign single arrays to Tom's columns:

% Time is first column:

Time=Sdid1(1:numTimeSteps,1);

% 2    Attitude Type Indicator, 3 = R,P,H
AttTypeInd=Sdid1(1:numTimeSteps,2);

% 3    GPS Type Indicator, 0 = no GPS data
GPSTypeInd=Sdid1(1:numTimeSteps,3);

% 4    North Accel, ft/sec^2
% 5    East Accel
% 6    Down Accel
NorthAccel=Sdid1(1:numTimeSteps,4);
EastAccel=Sdid1(1:numTimeSteps,5);
DownAccel=Sdid1(1:numTimeSteps,6);

% 7    North Velocity, ft/sec
% 8    East Velocity
% 9    Down Velocity
NorthVel=Sdid1(1:numTimeSteps,7);
EastVel=Sdid1(1:numTimeSteps,8);
DownVel=Sdid1(1:numTimeSteps,9);

%10    Latitude, deg
%11    Longitude, deg
%12    Altitude, ft
Lat=Sdid1(1:numTimeSteps,10);
Long=Sdid1(1:numTimeSteps,11);
Alt=Sdid1(1:numTimeSteps,12);

%13    X Body Rate, rad/sec
%14    Y Body Rate
%15    Z Body Rate
XBodyRate=Sdid1(1:numTimeSteps,13);
YBodyRate=Sdid1(1:numTimeSteps,14);
ZBodyRate=Sdid1(1:numTimeSteps,15);

%16    Roll, deg
%17    Pitch
%18    Heading
Roll=Sdid1(1:numTimeSteps,16);
Pitch=Sdid1(1:numTimeSteps,17);
Heading=Sdid1(1:numTimeSteps,18);

h1=figure;
set(h1,'Name','RMS Errors: Pos, Vel, Tilt');

% Plot Type indicators:
subplot(3,1,1), plot(Time,AttTypeInd,Time,GPSTypeInd);
subplot(3,1,1), title('   Attitude and GPS Type Indicators ');
subplot(3,1,1), xlabel(' Time (sec)');
subplot(3,1,1), ylabel('N/A');
legend('Att Type Ind','GPS Type Ind',2);

% Plot Acceleration:

subplot(3,1,2), plot(Time,NorthAccel,Time,EastAccel,Time,DownAccel);
subplot(3,1,2), title('   Accelerations ');
subplot(3,1,2), xlabel(' Time (sec)');
subplot(3,1,2), ylabel('ft/sec^2');

legend('North','East','Down',2);

% Plot Tilt errors:

subplot(3,1,3), plot(Time,NorthVel,Time,EastVel,Time,DownVel);
subplot(3,1,3), title('   Velocities ');
subplot(3,1,3), xlabel(' Time (sec)');
subplot(3,1,3), ylabel('ft/sec');
legend('North','East','Down',2);


h2=figure;
set(h2,'Name','RMS Errors: Other');

% Plot 50th Percentile errors:

subplot(3,1,1), plot(Time,Lat);
subplot(3,1,1), title('   Lat ');
subplot(3,1,1), xlabel(' Time (sec)');
subplot(3,1,1), ylabel('deg');
legend('Lat',2);

subplot(3,1,2), plot(Time,Long);
subplot(3,1,2), title('   Long ');
subplot(3,1,2), xlabel(' Time (sec)');
subplot(3,1,2), ylabel('deg');
legend('Long',2);

subplot(3,1,3), plot(Time,Alt);
subplot(3,1,3), title('  Alt ');
subplot(3,1,3), xlabel(' Time (sec)');
subplot(3,1,3), ylabel('ft');
legend('Alt',2);

% Plot angle errors:
h3=figure;
set(h2,'Name','Body Rates / Attitude');

subplot(3,1,2), plot(Time,XBodyRate,Time,YBodyRate,Time,ZBodyRate);
subplot(3,1,2), title('   Body Rates ');
subplot(3,1,2), xlabel(' Time (sec)');
subplot(3,1,2), ylabel('rad/sec');
legend('X','Y','Z',2);

% Plot attitude angles:

subplot(3,1,3), plot(Time,Roll,Time,Pitch,Time,Heading);
subplot(3,1,3), title('  Attitude ');
subplot(3,1,3), xlabel(' Time (sec)');
subplot(3,1,3), ylabel('deg');
legend('Roll','Pitch','Heading',2);
