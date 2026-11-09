function [omegaBbar]=compAvgBodyRates(Cbn,Cbnl, omega, time_step)

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
% 12/26/03:  Renamed omegaB to omega to reflect the fact that this
% quantity is an inertial rate, not a body rate.
% 06/12/04:  Replaced Segment and SegIndex with time_step.

% time_step  = Segment(SegIndex,2); % delta Tn
% Change 05/11/04 per T. De Vries suggestion:
Cbb = Cbn*Cbnl';

% Calculate approximate body angle changes.

p_x = .5 * ( Cbb(2,3)-Cbb(3,2) );
p_y = .5 * ( Cbb(3,1)-Cbb(1,3) );
p_z = .5 * ( Cbb(1,2)-Cbb(2,1) );

p = sqrt(p_x^2 + p_y^2 + p_z^2);
p_vec = [p_x; p_y; p_z];

sin_delta_theta = p;
cos_delta_theta = sqrt(1-p^2);
abs_delta_theta = atan2(sin_delta_theta,cos_delta_theta);

% Avoid divide by zero:
if sin_delta_theta ~=0
    avg_body_rate_earth = abs_delta_theta/sin_delta_theta * 1/time_step * p_vec;
else
    avg_body_rate_earth = 0;
end
% Change 5/11/04 to be consistent with T. De Vries suggestion:
omegaBbar = avg_body_rate_earth + Cbn*omega;

return;
