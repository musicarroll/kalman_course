% writeTrajFile.m
% This script writes a fixed width ASCII trajectory file in the De Vries format.
% The fields are:
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
%   19	          	True Heading		        radians		    2
%   20.			Platform Heading		radians.

% This script assumes that trajectory is in the Matlab workspace in the
% Trajectory array:

% Prompt user for filename:
filename = input('Enter filename for output: ','s');
fid = fopen(filename,'w');
format_str = '%+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e %+12.8e\n';
[m,n] = size(Trajectory);
for i=1:m
    fprintf(fid,format_str,Trajectory(i,:));
end
fclose(fid);
disp(['File ',filename,' written.']);
