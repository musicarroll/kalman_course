% tmcompare.m:  A script to compare Tom De Vries and Mike Carroll
% data.
% Load conversion factors
load 'convert.mat'
tom_filename = input('Enter name of Tom file:','s');
tomdata = load (tom_filename','-ascii');
% We assume that Tom's file has already been loaded into the workspace
% It may have a name like TEST_1G, but regardless, its structure is
% as follows:
% 1  Time, sec
% 2 and 3  Integer codes for the format of the attitude and GPS data 
% (ignore)
% 4, 5, 6  NED sensed acceleration, ft/sec^2
% 7, 8, 9  NED velocity, ft/sec
% 10  Latitude, rad
% 11  Longitude, rad
% 12  Altitude, ft
% 13, 14, 15  XYZ inertial body rates, rad/sec
% 16  Roll, deg
% 17  Pitch, deg
% 18  Heading, deg

% We also assume that Mike's trajectory has been run.  It produces
% an array called Trajectory.  Its structure is as
% following:
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
%   20              Platform Heading    radians

% To distinguish Mike's quantitities from Tom's, we will affix a 1 to
% Mike's items and 2 to Tom's.
% Thus, time is the following:
t1 = Trajectory(:,1);
t2 = tomdata(:,1);

% Acceleration Comparison:
an1=Trajectory(:,2);
an2=tomdata(:,4); % in ft/sec^2
an2 = an2*ft2m; % now in m/sec^2
delta_an = an1-an2;
max_delta_an = max(abs(delta_an));

ae1=Trajectory(:,3);
ae2=tomdata(:,5); % in ft/sec^2
ae2 = ae2*ft2m; % now in m/sec^2
delta_ae = ae1-ae2;
max_delta_ae = max(abs(delta_ae));

ad1=Trajectory(:,4);
ad2=tomdata(:,6); % in ft/sec^2
ad2 = ad2*ft2m; % now in m/sec^2
delta_ad = ad1-ad2;
max_delta_ad = max(abs(delta_ad));

figure;
title(['Acceleration Compare.  m/sec^2']);
subplot(3,3,1), plot(t1,delta_an);
legend(['Delta accel north. Max delta north=',num2str(max_delta_an)]);
subplot(3,3,2), plot(t1,delta_ae);
legend(['Delta accel east. Max delta east=',num2str(max_delta_ae)]);
subplot(3,3,3), plot(t1,delta_ad);
legend(['Delta accel down. Max delta down=',num2str(max_delta_ad)]);

subplot(3,3,4), plot(t1,an1);
legend('North Accel (Carroll)');
subplot(3,3,5), plot(t1,ae1);
legend('East Accel (Carroll)');
subplot(3,3,6), plot(t1,ad1);
legend('Down Accel (Carroll)');

subplot(3,3,7), plot(t1,an2);
legend('North Accel (DeVries)');
subplot(3,3,8), plot(t1,ae2);
legend('East Accel (DeVries)');
subplot(3,3,9), plot(t1,ad2);
legend('Down Accel (DeVries)');


% Velocity Comparison:
vn1=Trajectory(:,5);
vn2=tomdata(:,7); % in ft/sec^2
vn2 = vn2*ft2m; % now in m/sec^2
delta_vn = vn1-vn2;
max_delta_vn = max(abs(delta_vn));

ve1=Trajectory(:,6);
ve2=tomdata(:,8); % in ft/sec^2
ve2 = ve2*ft2m; % now in m/sec^2
delta_ve = ve1-ve2;
max_delta_ve = max(abs(delta_ve));

vd1=Trajectory(:,7);
vd2=tomdata(:,9); % in ft/sec^2
vd2 = vd2*ft2m; % now in m/sec^2
delta_vd = vd1-vd2;
max_delta_vd = max(abs(delta_vd));

figure;
title(['Velocity Compare.  m/sec']);
subplot(3,3,1), plot(t1,delta_vn);
legend(['Delta vel north. Max delta north=',num2str(max_delta_vn)]);
subplot(3,3,2), plot(t1,delta_ve);
legend(['Delta vel east. Max delta east=',num2str(max_delta_ve)]);
subplot(3,3,3), plot(t1,delta_vd);
legend(['Delta vel down. Max delta down=',num2str(max_delta_vd)]);

subplot(3,3,4), plot(t1,vn1);
legend('North Velocity (Carroll)');
subplot(3,3,5), plot(t1,ve1);
legend('East Velocity (Carroll)');
subplot(3,3,6), plot(t1,vd1);
legend('Down Velocity (Carroll)');

subplot(3,3,7), plot(t1,vn2);
legend('North Velocity (DeVries)');
subplot(3,3,8), plot(t1,ve2);
legend('East Velocity (DeVries)');
subplot(3,3,9), plot(t1,vd2);
legend('Down Velocity (DeVries)');

% Lat/Lon/Alt Comparison:
lat1=Trajectory(:,11)*180/pi;
lat2=tomdata(:,10); % in rad
lat2 = lat2*180/pi; % now in deg
delta_lat = (lat1-lat2)*3600; % in arc secs
max_delta_lat = max(abs(delta_lat));

lon1=Trajectory(:,12)*180/pi;
lon2=tomdata(:,11); % in rad
lon2 = lon2*180/pi; % now in deg
delta_lon = (lon1-lon2)*3600;
max_delta_lon = max(abs(delta_lon));

alt1=Trajectory(:,13);
alt2=tomdata(:,12); % in ft
alt2 = alt2*ft2m; % now in m
delta_alt = alt1-alt2;
max_delta_alt = max(abs(delta_alt));

figure;
title(['Position Compare.  Lat/Lon/Alt in Deg/Deg/m']);
subplot(3,3,1), plot(t1,delta_lat);
legend(['Delta lat (arc sec). Max delta lat=',num2str(max_delta_lat)]);
subplot(3,3,2), plot(t1,delta_lon);
legend(['Delta longitude. (arc sec) Max delta lon=',num2str(max_delta_lon)]);
subplot(3,3,3), plot(t1,delta_alt);
legend(['Delta altitude. Max delta alt=',num2str(max_delta_alt)]);

subplot(3,3,4), plot(t1,lat1);
legend('Latitude (Carroll)');
subplot(3,3,5), plot(t1,lon1);
legend('Longitude (Carroll)');
subplot(3,3,6), plot(t1,alt1);
legend('Altitude (Carroll)');

subplot(3,3,7), plot(t1,lat2);
legend('Latitude (DeVries)');
subplot(3,3,8), plot(t1,lon2);
legend('Longitude (DeVries)');
subplot(3,3,9), plot(t1,alt2);
legend('Altitude (DeVries)');

% Roll/Pitch/Heading Comparison:
roll1=Trajectory(:,17)*180/pi;
roll2=tomdata(:,16); % in deg
delta_roll = (roll1-roll2)*3600; % in arc secs
max_delta_roll = max(abs(delta_roll));

pitch1=Trajectory(:,18)*180/pi;
pitch2=tomdata(:,17); % in deg
delta_pitch = (pitch1-pitch2)*3600;
max_delta_pitch = max(abs(delta_pitch));

hdg1=Trajectory(:,19)*180/pi;
hdg2=tomdata(:,18); % in ft
delta_hdg = (hdg1-hdg2)*3600; % delta in arc sec
max_delta_hdg = max(abs(delta_hdg));

figure;
title(['Position Compare.  Lat/Lon/Alt in Deg/Deg/m']);
subplot(3,3,1), plot(t1,delta_roll);
legend(['Delta roll (arc sec). Max delta roll=',num2str(max_delta_roll)]);
subplot(3,3,2), plot(t1,delta_pitch);
legend(['Delta pitch. (arc sec) Max delta =',num2str(max_delta_pitch)]);
subplot(3,3,3), plot(t1,delta_hdg);
legend(['Delta heading. Max delta heading=',num2str(max_delta_hdg)]);

subplot(3,3,4), plot(t1,roll1);
legend('Roll (Carroll)');
subplot(3,3,5), plot(t1,pitch1);
legend('Pitch (Carroll)');
subplot(3,3,6), plot(t1,hdg1);
legend('Heading (Carroll)');

subplot(3,3,7), plot(t1,roll2);
legend('Roll (DeVries)');
subplot(3,3,8), plot(t1,pitch2);
legend('Pitch (DeVries)');
subplot(3,3,9), plot(t1,hdg2);
legend('Heading (DeVries)');












