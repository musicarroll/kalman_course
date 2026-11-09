function delta_R = pos_deriv(time_step,tilt,pos_err,vel_err,vel_integ,rel_rate_integ)
  delta_R = [0;0;0];
  delta_R(1) = time_step*vel_err(1)+vel_integ(2)*tilt(3)-vel_integ(3)*tilt(2)-rel_rate_integ(2)*pos_err(3);
  delta_R(2) = time_step*vel_err(2)+vel_integ(3)*tilt(1)-vel_integ(1)*tilt(3)+rel_rate_integ(1)*pos_err(3);
  delta_R(3) = time_step*vel_err(3)+vel_integ(1)*tilt(2)-vel_integ(2)*tilt(1)+rel_rate_integ(2)*pos_err(1)-rel_rate_integ(1)*pos_err(2);
  