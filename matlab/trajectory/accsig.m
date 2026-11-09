% Plots Tom De Vries RMS Error Files
% Revision Histoy:
% Date      Author      Description
% 04/09/02  M. Carroll  Initial Release

% Assumes Accsig array has been created through import of data from one
% of Tom's data files with 16 columns.

[numTimeSteps,numCols] = size(Accsig);

% Assign single arrays to Tom's columns:

% Time is first column:

Time=Accsig(1:numTimeSteps,1);

% Pos errors are next 3 columns (in feet)
AccelBiasX=Accsig(1:numTimeSteps,2);
AccelBiasY=Accsig(1:numTimeSteps,3);
AccelBiasZ=Accsig(1:numTimeSteps,4);

% Vel errors are next 3 columns (in fps)
AccelMarkovX=Accsig(1:numTimeSteps,5);
AccelMarkovY=Accsig(1:numTimeSteps,6);
AccelMarkovZ=Accsig(1:numTimeSteps,7);

% Tilt errors are next 3 columns (in fps)
AccelSFX=Accsig(1:numTimeSteps,8);
AccelSFY=Accsig(1:numTimeSteps,9);
AccelSFZ=Accsig(1:numTimeSteps,10);

h1=figure;
set(h1,'Name','Bias and Random Walk');

% Plot Position errors:

subplot(3,1,1), plot(Time,AccelBiasX,Time,AccelBiasY,Time,AccelBiasZ);
subplot(3,1,1), title('   Accel Bias ');
subplot(3,1,1), xlabel(' Time (sec)');
subplot(3,1,1), ylabel('mg');
legend('Accel Bias X','Accel Bias Y','Accel Bias Z',2);

% Plot Velocity errors:

subplot(3,1,2), plot(Time,AccelMarkovX,Time,AccelMarkovY,Time,AccelMarkovZ);
subplot(3,1,2), title('   Accel Markov Random Walk ');
subplot(3,1,2), xlabel(' Time (sec)');
subplot(3,1,2), ylabel('mg');
legend('Accel Markov X','Accel Markov Y','Accel Markov Z',2);

% Plot Tilt errors:

subplot(3,1,3), plot(Time,AccelSFX,Time,AccelSFY,Time,AccelSFZ);
subplot(3,1,3), title('   Accel Scale Factors ');
subplot(3,1,3), xlabel(' Time (sec)');
subplot(3,1,3), ylabel('ppm');
legend('Accel SF X ','Accel SF Y','Accel SF Z',2);

% Process other errors.

% Cross-axis scale factor errors are next 6 columns (in feet)
AccelSFXY=Accsig(1:numTimeSteps,11);
AccelSFXZ=Accsig(1:numTimeSteps,12);
AccelSFYX=Accsig(1:numTimeSteps,13);
AccelSFYZ=Accsig(1:numTimeSteps,14);
AccelSFZX=Accsig(1:numTimeSteps,15);
AccelSFZY=Accsig(1:numTimeSteps,16);

h2=figure;
set(h2,'Name','Misalignments');

% X cross-axis SF errors:

subplot(3,1,1), plot(Time,AccelSFXY,Time,AccelSFXZ);
subplot(3,1,1), title(' Accel XY, XZ Misalignment ');
subplot(3,1,1), xlabel(' Time (sec)');
subplot(3,1,1), ylabel('urad');
legend('XY Misalign','XZ Misalign',2);

% Y cross-axis SF errors:

subplot(3,1,2), plot(Time,AccelSFYX,Time,AccelSFYZ);
subplot(3,1,2), title(' Accel YX, YZ Misalignment ');
subplot(3,1,2), xlabel(' Time (sec)');
subplot(3,1,2), ylabel('urad');
legend('YX Misalign','YZ Misalign',2);

% Z cross-axis SF errors:

subplot(3,1,3), plot(Time,AccelSFZX,Time,AccelSFZY);
subplot(3,1,3), title(' Accel ZX, ZY Misalign ');
subplot(3,1,3), xlabel(' Time (sec)');
subplot(3,1,3), ylabel('urad');
legend('ZX Misalign','ZY Misalign',2);
