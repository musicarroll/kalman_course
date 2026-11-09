% straperr_drv.m
% This script runs the straperr.mdl simulation using a trajectory output by genTraj.m.



% Initialize lat/lon DCM matrix:
% Note: lat0,lon0,alt0 come from genTraj.m
D0 = initPosDCM(lat0,lon0,alt0,'deg');

% Run the simulation:
TStart = 0;
TFinal = T-1; % from genTraj.m

% Load data into top-level inports for straperr.mdl:
TIn = TStart:TFinal;
TIn = TIn';


% Array with time to drive the gyro and accelerometer Inports in straperr.mdl:
% Note body_rate_* and body_accel_* come from genTraj.m

tu = [TIn, body_rate_x, body_rate_y, body_rate_z, ...
        body_accel_x, body_accel_y, body_accel_z];

% Execute the model straperr.mdl using the sim command:
[t,x,DCM_L2E,accel_L,altitude,v_L,E_L,platform_angle_error,v_err_L] ...
  = sim('straperr',[TStart TFinal],[],tu);

% Plot the results.
len=length(t);
% Note: Using element-by-element assignment since vector assignment
% such as vec1 = Cov1(1,1,:) yields a structure of size 1x1x11.
% And such structures are abhorrent to the plot function. :-)
for i=1:len
% elements of E_L matrix:
    temp = skew2vec(E_L(:,:,i));
    e_Lx(i) = temp(1);
    e_Ly(i) = temp(2);
    e_Lz(i) = temp(3);
% elements of v_L vector:
    v_Lx(i) = v_L(1,1,i);
    v_Ly(i) = v_L(2,1,i);
    v_Lz(i) = v_L(3,1,i);
% elements of v_err_L vector:
    v_err_Lx(i) = v_err_L(1,1,i);
    v_err_Ly(i) = v_err_L(2,1,i);
    v_err_Lz(i) = v_err_L(3,1,i);

end
figure;
plot(t,e_Lx,t,e_Ly,t,e_Lz);
legend('Pos Err X','Pos Err Y','Pos Err Z');
title('Position Errors in Local Level Frame. Strapdown Error Model (straperr.mdl) Simulation.');

figure;
plot(t,v_err_Lx,t,v_err_Ly,t,v_err_Lz);
legend('Vel Err X','Vel Err Y','Vel Err Z');
title('Velocity Errors in Local Level Frame. Strapdown Error Model (straperr.mdl) Simulation.');

figure;
plot(t,v_Lx,t,v_Ly,t,v_Lz);
legend('Vel X','Vel Y','Vel Z');
title('Velocity in Local Level Frame. Strapdown Error Model (straperr.mdl) Simulation.');
