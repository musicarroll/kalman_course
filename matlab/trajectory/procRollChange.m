function [roll] = procRollChange(roll, cmd_roll, time_step, roll_rate_limit, g, Vh)
% procRollChange.m processes roll changes
% as outlined in section 6.3:
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
% 06/14/04:  Replaced Segment and SegIndex with time_step.

%time_step  = Segment(SegIndex,2);
delta_cmd_roll = cmd_roll - roll; % required roll change 

if delta_cmd_roll~= 0 % test for non-zero change
	delta_phi_k = sign(delta_cmd_roll)*roll_rate_limit*time_step;
	if abs(delta_cmd_roll) < abs(delta_phi_k)
		delta_phi_k = delta_cmd_roll;
	end
	roll = roll + delta_phi_k;
end
return;
    
    
