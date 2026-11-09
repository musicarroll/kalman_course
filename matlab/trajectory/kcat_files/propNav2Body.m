function [Cbn, Cbnl, alpha, true_hdg] = propNav2Body(Cbn, alpha, alpha_dot, hdg, pitch, roll, time_step)
% Change Log:
% 03/26/04: Added alpha as a return value.  It was being
% updated but its change not known or used outside of this routine.  Could
% be cause for the heading being different from Tom DeVries' heading.
% 04/02/04: Added true_hdg as return value.
% 06/12/04: Replaced Segment and SegIndex as input arguments with time_step.

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


Cbnl = Cbn;  % store previous transformation

alpha = alpha + alpha_dot*time_step;  % extrapolate wander angle

true_hdg = hdg + alpha; % compute true heading

Cbn = euler2DCM (roll,pitch,true_hdg);

return;
