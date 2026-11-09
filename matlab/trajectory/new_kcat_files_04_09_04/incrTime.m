function [status,T,SegIndex] = incrTime(T,SegIndex,Segment,num_segs)
% This function implements the time logic indicated
% section 6.1.
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

time_step = Segment(SegIndex,2);

T = T + time_step; %boost time by time step

%  Assume we are not yet finished with the simulation:
status = 0;

if T > Segment(SegIndex,1) % test for end of segment
    if SegIndex <  num_segs 	%	test for more segments
		SegIndex = SegIndex + 1	%boost segment index
    else
        status = 1; % end of trajectory
    end
end % end of segment test

