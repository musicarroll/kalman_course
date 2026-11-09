function [V] = procVelocityChange(V,time_step, Vn, Axn, g, Vh)
% procVelocityChange.m processes V changes
% as outlined in section 6.5:
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
% 06/14/04:  Replaced Segment and SegIndex with time_step, Vn, and Axn.

%time_step  = Segment(SegIndex,2); % delta Tn
%Vn = Segment(SegIndex,5); % Final total velocity
%Axn = Segment(SegIndex,6); % Longitudinal acceleration

delta_V = Vn - V; % required V change

if delta_V~= 0 % test for non-zero velocity change
	Ax = sign(delta_V)*Axn; % specified acceleration
	delta_V_k = Ax*time_step; % specified velocity change
	if abs(delta_V) < abs(delta_V_k) % test for last time step
		delta_V_k = delta_V; % last velocity change
	end
	V = V + delta_V_k;
end

return;



