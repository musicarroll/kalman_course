function [avg] = dyn_recursive_avg(meas,gain,prior_est,samp_time,slope)
  slope = 5;  % This should match constant in input generator.
  avg = zeros(1,1);
  % Note how we use the slope to extrapolate the last 
  % estimate:
  extrap_est = prior_est + slope*samp_time;
  avg = prior_est + gain*(meas-extrap_est);
return;
