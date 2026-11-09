function [Vned, avg_vel, avg_vel_rate] = compVelocity(Vned, V, Cbn, time_step)

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
% Change Log
% Replace Segment and SegIndex with time_step.

Vned_last = Vned;  % store previous velocities
% Note the error in De Vries section 6.9 where Cbn is used instead
% of Cbn':
Vned = Cbn*[V;0;0]; % trajectory navigation velocities
avg_vel = (Vned+Vned_last)/2; % average velocities
avg_vel_rate = (Vned - Vned_last)/time_step;  % average velocity rate

return;
