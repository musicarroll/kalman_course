% Plots Tom De Vries RMS Error Files
% Revision Histoy:
% Date      Author      Description
% 04/09/02  M. Carroll  Initial Release

% Assumes Gyrosig array has been created through import of data from one
% of Tom's data files with 16 columns.

[numTimeSteps,numCols] = size(Gyrosig);

% Assign single arrays to Tom's columns:

% Time is first column:

Time=Gyrosig(1:numTimeSteps,1);

% Pos errors are next 3 columns (in feet)
GyroBiasX=Gyrosig(1:numTimeSteps,2);
GyroBiasY=Gyrosig(1:numTimeSteps,3);
GyroBiasZ=Gyrosig(1:numTimeSteps,4);

% Vel errors are next 3 columns (in fps)
GyroMarkovX=Gyrosig(1:numTimeSteps,5);
GyroMarkovY=Gyrosig(1:numTimeSteps,6);
GyroMarkovZ=Gyrosig(1:numTimeSteps,7);

% Tilt errors are next 3 columns (in fps)
GyroSFX=Gyrosig(1:numTimeSteps,8);
GyroSFY=Gyrosig(1:numTimeSteps,9);
GyroSFZ=Gyrosig(1:numTimeSteps,10);

h1=figure;
set(h1,'Name','Bias and Random Walk');

% Plot Position errors:

subplot(3,1,1), plot(Time,GyroBiasX,Time,GyroBiasY,Time,GyroBiasZ);
subplot(3,1,1), title('   Gyro Bias ');
subplot(3,1,1), xlabel(' Time (sec)');
subplot(3,1,1), ylabel('dph');
legend('Gyro Bias X','Gyro Bias Y','Gyro Bias Z',2);

% Plot Velocity errors:

subplot(3,1,2), plot(Time,GyroMarkovX,Time,GyroMarkovY,Time,GyroMarkovZ);
subplot(3,1,2), title('   Gyro Markov Random Walk ');
subplot(3,1,2), xlabel(' Time (sec)');
subplot(3,1,2), ylabel('dph');
legend('Gyro Markov X','Gyro Markov Y','Gyro Markov Z',2);

% Plot Tilt errors:

subplot(3,1,3), plot(Time,GyroSFX,Time,GyroSFY,Time,GyroSFZ);
subplot(3,1,3), title('   Gyro Scale Factors ');
subplot(3,1,3), xlabel(' Time (sec)');
subplot(3,1,3), ylabel('ppm');
legend('Gyro SF X ','Gyro SF Y','Gyro SF Z',2);

% Process other errors.

% Cross-axis scale factor errors are next 6 columns (in feet)
GyroSFXY=Gyrosig(1:numTimeSteps,11);
GyroSFXZ=Gyrosig(1:numTimeSteps,12);
GyroSFYX=Gyrosig(1:numTimeSteps,13);
GyroSFYZ=Gyrosig(1:numTimeSteps,14);
GyroSFZX=Gyrosig(1:numTimeSteps,15);
GyroSFZY=Gyrosig(1:numTimeSteps,16);

h2=figure;
set(h2,'Name','Misalignments');

% X cross-axis SF errors:

subplot(3,1,1), plot(Time,GyroSFXY,Time,GyroSFXZ);
subplot(3,1,1), title(' Gyro XY, XZ Misalignment ');
subplot(3,1,1), xlabel(' Time (sec)');
subplot(3,1,1), ylabel('urad');
legend('XY Misalign','XZ Misalign',2);

% Y cross-axis SF errors:

subplot(3,1,2), plot(Time,GyroSFYX,Time,GyroSFYZ);
subplot(3,1,2), title(' Gyro YX, YZ Misalignment ');
subplot(3,1,2), xlabel(' Time (sec)');
subplot(3,1,2), ylabel('urad');
legend('YX Misalign','YZ Misalign',2);

% Z cross-axis SF errors:

subplot(3,1,3), plot(Time,GyroSFZX,Time,GyroSFZY);
subplot(3,1,3), title(' Gyro ZX, ZY Misalign ');
subplot(3,1,3), xlabel(' Time (sec)');
subplot(3,1,3), ylabel('urad');
legend('ZX Misalign','ZY Misalign',2);
