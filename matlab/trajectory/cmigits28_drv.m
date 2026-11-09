% cmigits28_drv.m
% Change History:
% 05/14/04:  Initial Version
% 05/20/04:  Created build28StateTransMat function
% 07/29/04:  Removed dependency on T
% This script runs the KCAT-TV simulation using the
% C-MIGITS III 28-state inertial error dynamics model.


% Second, load the various model matrix parameters.
% This set includes P0, Q, R, and H.

load('cmigits28_model_01.mat');

% The values for F
% are taken successively from the 9x9xT array
% F_array, where T is number of time steps (1Hz assumption)
% The array 3-D F_array should be
% obtained from the workspace by a call to
% build28StatetransMat(Trajectory,RelRates);


% Run the simulation:
TStart = 0;
[m,n]=size(Segment);
TFinal = Segment(m,1); % from genTraj.m

TIn = TStart:TFinal;
TIn = TIn';


% Array with time to drive

% First build Phi matrix array from genTraj.m outputs:
Phi_array = build28StateTransMat(Trajectory,RelRates);
% We need to build a Matlab "Structure with Time".
% Simulink can read data from the workspace in the
% form of a structure whose name is specified in the
% input text field of the Workspace I/O pane.  The
% Workspace I/O pane is the second tab on the Simulation
% Parameters dialogue box.
% The input structure must have
% two top-level fields: 'time' and 'signals'.  The 'time'
% field contains a column vector of the simulation
% times.  The 'signals' field contains an array of
% substructures, each of which corresponds to a
% model input port.  Thus, signals(1) is a substructure
% associated with inport 1, signals(2) is associated with
% inport 2, etc.
%
% Each 'signals' substructure must contain two further fields, one
% named 'values' and another named 'dimensions'.  The 'values'
% field must contain an array of inputs for the corresponding
% inport indexed by the time field.  The 'dimensions' field
% specifies the dimensions of the input.  Since our input is
% a stream 2-D system dynamics matrices, the 'dimensions' field
% is a two-element vector indicating the number of rows and
% columns of each matrix, i.e,. [28,28].

% Since the inputs for port 1 are matrices, the 'values' field must
% be 3-D array of 28x28xT real numbrs.  T is the number of time points.
% This is why the function build9StateSysDyn(Trajectory,RelRates)
% built F_array as a 28x28xT 3-D array.

% If further inports are later created for KCAT-TV, then their input
% values would also have to be incorporated into this structure with time.

% Our structure with time is called inStruct for "input structure."
% You must ensure that the input check box is checked on the Workspace I/O
% pane. And that the structure name inStruct is given in the field to
% to the right of the check box.

% Also VERY IMPORTANT!!!: You must make sure that the inport into which the
% the inStruct structure is fed has the appropriate dimensions specified in
% the port dimensions field in the Block Parameters dialogue.  There appears
% to be a bug in Simulink in that a -1 specified makes
% Simulink think that inStruct
% is a 1x1 array.  You must specify [28 28] here instead.  Then all is well.
% I could not find anything in the help documentation about this!!!

inStruct.time = TIn;
inStruct.signals(1).values = Phi_array;
inStruct.signals(1).dimensions = [28,28];

[t,x,Cov1,Cov2,Kalgain,Phi,...
  n_pos_err_cov,e_pos_err_cov,d_pos_err_cov,...
  n_vel_err_cov,e_vel_err_cov,d_vel_err_cov,...
  n_tilt_cov,e_tilt_cov,az_err_cov] ...
  = sim('kcat_tv',[TStart TFinal],[],inStruct);

% Plot the results.

figure;
len=length(t);
% Note: Using element-by-element assignment since vector assignment
% such as vec1 = Cov1(1,1,:) yields a structure of size 1x1x11.
% And such structures are abhorrent to the plot function. :-)
for i=1:len
% Diagonal elements of covariance matrix prior to update:
    vec1(i) = Cov1(1,1,i);
    vec2(i) = Cov1(2,2,i);
% Diagonal elements of covariance matrix after update:
    vec3(i) = Cov2(1,1,i);
    vec4(i) = Cov2(2,2,i);
% Kalman gains:
    k1(i)=Kalgain(1,1,i);
    k2(i)=Kalgain(2,1,i);
end
subplot(2,1,1), plot(t,vec1,t,vec3)
legend('P(1,1)(-)','P(1,1)(+)')
title('KCAT Simulation. First Two Diagonal Elements of Error Covariance Matrix (before and after updates).\newline Canonical 9-State Inertial Error Dynamics Model');
subplot(2,1,2), plot(t,vec2,t,vec4);
legend('P(2,2)(-)','P(2,2)(+)');

figure;
plot(t,k1,t,k2);
title('KCAT Simulation.  Kalman Gains:  Canonical 9-State Inertial Error Dynamics Model');
legend('Position Gain','Velocity Gain');

% If you run the KCAT simulation interactively within Simulink, and
% both tout and yout are generated automatically
% when the appropriate boxes
% are checked in the Workspace I/O Pane.  Then following can be run:
% simplot(yout,[5,6,7,8,9,10,11,12]);
% This generates a plot of the covariance diagonals.

% The following figure plots all 9 covariances in a 3x3 figure:
% To do:  Create similar figures for gains and transition matrices
% More to do:  Splice prior and post Covariances together to get
% a typical sawtooth pattern.

figure;
title('Estimation Error Variances (Diagonal Elements of Covarance Matrix)');
subplot(3,3,1), plot(t,n_pos_err_cov(:)); % Note the trick of using the colon
legend('X Pos Err Cov');
ylabel('m^2');
subplot(3,3,4), plot(t,e_pos_err_cov(:));
legend('Y Pos Err Cov');
subplot(3,3,7), plot(t,d_pos_err_cov(:));
legend('Z Pos Err Cov');

subplot(3,3,2), plot(t,n_vel_err_cov(:));
legend('X Vel Err Cov');
subplot(3,3,5), plot(t,e_vel_err_cov(:));
legend('Y Vel Err Cov');
subplot(3,3,8), plot(t,d_vel_err_cov(:));
legend('Z Vel Err Cov');

subplot(3,3,3), plot(t,n_tilt_cov(:));
legend('X Tilt Cov');
subplot(3,3,6), plot(t,e_tilt_cov(:));
legend('Y Tilt Cov');
subplot(3,3,9), plot(t,az_err_cov(:));
legend('Az Err Cov');


disp('End KCAT Simulation.');
