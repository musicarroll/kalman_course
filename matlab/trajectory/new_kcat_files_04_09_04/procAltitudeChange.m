function [alt, Av] = procAltitudeChange(alt,time_step, final_alt, alt_rate, Avn, pitch_angle, pitch_rate_limit,g, V, Vv)
% procAltitudeChange.m processes Altitude changes
% as outlined in section 6.2:
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
% 06/14/04:  Replaced Segment and SegIndex with time_step, final_alt, alt_rate, Avn, pitch_angle.
%time_step   = Segment(SegIndex,2);
%final_alt   = Segment(SegIndex,7);
delta_alt   = final_alt - alt; % required Altitude change
%alt_rate    = Segment(SegIndex,8);
%Avn         = Segment(SegIndex,9);
%pitch_angle = Segment(SegIndex,10);
delta_h_k   = 0;  %default value
Av          = 0;  %default value

if delta_alt ~= 0 % test for non-zero change
   	Av = g* Avn;
	if pitch_angle == 0
		Vvn = sign(delta_alt)*abs(alt_rate);
	else
		Vvn = V*sin(pitch_angle);
		Avmax = V * pitch_rate_limit;
		if Av > Avmax
			Av = Avmax;
		end
	end

	Av = sign(delta_alt)*Av;
	decel_dist = Vvn^2 / 2 * Av;

	if abs(decel_dist) >=abs(delta_alt)
		Vvn = 0;
		decel_time = 2*delta_alt / Vv;
		Av = -Vv/decel_time;
	else
		decel_time = 2 *time_step;
	end

	delta_vert_vel = Vvn - Vv;

	if delta_vert_vel ~=0
		if decel_time > time_step
			delta_vert_vel_k = Av*time_step;
			delta_h_k = (Vv+delta_vert_vel_k/2) * time_step;
		else
			delta_vert_vel_k = delta_vert_vel;
			delta_h_k = delta_alt;
		end
		Vv = Vv + delta_vert_vel_k;
	end

	alt = alt + delta_h_k;
end
return;



