function [hdg, cmd_roll] = procHeadingChange(hdg,time_step, final_hdg, turn_accel,hdg_rate_limit,g, Vh)
% procHeadingChange.m processes heading changes
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
% 06/12/04:  Replaced Segment and SegIndex with time_step, final_hdg, turn_accel

%time_step  = Segment(SegIndex,2);
%final_hdg  = Segment(SegIndex,3);
delta_hdg  = final_hdg - hdg; % required heading change
cmd_roll   = 0;  % default value

%turn_accel = Segment(SegIndex,4);

if delta_hdg ~= 0 % test for non-zero change
    if turn_accel == 0 % turn acceleration not specified
        delta_psi_k = sign(delta_hdg)*hdg_rate_limit*time_step; % flat turn at specified rate
        cmd_roll = 0; % commanded roll = 0
    else % turn acceleration specified
        delta_psi_k = sign(delta_hdg)*abs(turn_accel)*time_step/Vh; % heading change at specified g level
        if turn_accel > 0 % test for positive turn acceleration
            cmd_roll = sign(delta_hdg)*atan(turn_accel); % coordinated turn roll angle
        else % flat turn
            cmd_roll = 0;
        end % end of turn acceleration test
    end % end of turn type test
    
    if abs(delta_hdg) < abs(delta_psi_k) % test for end of turn
        delta_psi_k = delta_hdg; % complete turn
        cmd_roll = 0; % set commanded roll to zero
    end
    
    hdg = hdg + delta_psi_k; % extrapolate heading
end
return;

    
    
