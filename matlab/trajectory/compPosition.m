function [POSecef, POSgeo] = compPosition(POSgeo, avg_vel, Rlat, Rlon, time_step)

% Each row of Segment contains:
% 1.   Final time
% 2.   Time step
% 3.   Final heading
% 4.   Centripital turn acceleration (g's)
% 5.   Final total velocity
% 6.   Longitudinal acceleration (g's)
% 7.   Final altitude
% 8.   Altitude rate  
% 9.   Vertical acceleration (g's)
% 10.  Pitch angle (climb or dive)   
% Change Log:
% 06/12/04:  Replaced Segment and SegIndex with time_step.

% time_step  = Segment(SegIndex,2); % delta Tn
lat = POSgeo(1);
lon = POSgeo(2);
alt = POSgeo(3);
Van = avg_vel(1);
Vae = avg_vel(2);

% Note original equation in De Vrees spec was in error. Had subtraction of
% increment instead of addition:
lat = lat + Van*time_step/Rlat; % extrapolate latitude (is this correct?  should it be Rlat?  Yes, it should be Rlat)

lon = lon + Vae*time_step/(Rlon*cos(lat));

POSgeo = [lat;lon;alt];

[Rlat, Rlon, ae, D] = getRadii(lat,alt);  % calculate new radii
    
POSecef = zeros(3,1);
POSecef(1) = (ae/D + alt)*cos(lat)*cos(lon);
POSecef(2) = (ae/D+alt)*cos(lat)*sin(lon);
POSecef(3) = Rlat*sin(lat);

return;
