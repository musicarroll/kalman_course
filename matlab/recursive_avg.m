function [avg] = recursive_avg(meas,gain,prior_est)
  avg = zeros(1,1);
  avg = prior_est + gain*(meas-prior_est);
return;