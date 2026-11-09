% Plots RMS Error Files
% Revision History:
% Date      Author      Description
% 04/09/02  M. Carroll  Initial Release
% 06/24/03  M. Carroll  Revised per new RMS file formats from TDV.
% 08/07/03  M. Carroll  Revised plot markers, font, and width per P. Carter.

% Assumes Navsig array has been created through import of data from one
% of data files with 21 columns.
% Format:
%Item	Content

%  1	  Time, sec 
%  2	  x Position, ft 	
%  3	  y Position, ft
%  4      z Position, ft
%  5      x Velocity, ft/sec
%  6      y Velocity, ft/sec
%  7      z Velocity, ft/sec
%  8      x Tilt, mrad
%  9      y Tilt, mrad
% 10      z Tilt or Heading, mrad
% 11      Level Position CEP, ft
% 12      Total Position SEP, ft
% 13      X Body Velocity, ft/sec
% 14      Y Body Velocity, ft/sec
% 15      Z Body Velocity, ft/sec
% 16      Computed Angle of Attack, mrad
% 17      Computed Drift Angle, mrad
% 18      Position along velocity vector, ft
% 19      Level Position in plane normal to velocity vector, ft
% 20      Vertical Position in plane normal to velocity vector, ft
% 21      Position CEP in plane normal to velocity vector, ft

% x, y, z are nominally east, north, up Navigation axes

clear;
showandtell=0;
f2nm = 1/6076.1;
R='';
    fn='';
    disp(' ');
    disp('Enter Drive:\Path from which to load the Navsig (De Vries) data');
    disp(' ');
    disp('     {example: c:\matlab\work\navsig');
    disp(' ');
    R=input('?   ','s'); % Note: R contains directory path to files of interest.
    disp(' ');
    file=[R];
    if exist(file)
        disp('Loading Navsig data files');
    else
        disp('File do not exist.  Exiting ...');
        disp(R);
        return
    end
    
%disp('Enter filename:');
%filename = input('?','s');
filename='navsig.dat';
fn = [R,'\',filename];
Navsig=load(fn,'-ascii');

filename='accsig.dat';
fn = [R,'\',filename];
Accsig=load(fn,'-ascii');

filename='gyrosig.dat';
fn = [R,'\',filename];
Gyrosig=load(fn,'-ascii');

strTitle = input('Enter additional title info: ','s');

[numTimeSteps,numCols] = size(Navsig);
disp(['Number of Time Steps in data: ',num2str(numTimeSteps)]);
userTimeSteps = input('Enter number of time steps to plot (<CR> plots all):');
if isempty(userTimeSteps)
    numToPlot = numTimeSteps;
else
    numToPlot = userTimeSteps;
end


% Assign single arrays to Tom's columns:

% Time is first column:

Time=Navsig(1:numToPlot,1);

gpsLossTime = input('Enter time point at which GPS is lost: ');
gpsLossIndices = find(Time==gpsLossTime);
gpsLossIndex = gpsLossIndices(1);  % Use first returned index.
gpsFiveMinuteIndices = find(Time==(gpsLossTime+300));
gpsFiveMinuteIndex = gpsFiveMinuteIndices(1);
display(['Time of 5 min point after GPS Loss: ',num2str(Time(gpsFiveMinuteIndex))]);
display(['Index of GPS Loss point: ',num2str(gpsLossIndex)]);

% Pos errors are next 3 columns (in feet)
Rx=Navsig(1:numToPlot,2);
Ry=Navsig(1:numToPlot,3);
Rz=Navsig(1:numToPlot,4);

% Vel errors are next 3 columns (in fps)
Vx=Navsig(1:numToPlot,5);
Vy=Navsig(1:numToPlot,6);
Vz=Navsig(1:numToPlot,7);

% Tilt errors are next 3 columns (in fps)
Phix=Navsig(1:numToPlot,8);
Phiy=Navsig(1:numToPlot,9);
Psiz=Navsig(1:numToPlot,10);

% 50th Percentile errors are next 2 columns (in feet)
LevPosCEP=Navsig(1:numToPlot,11);
TotPosCEP=Navsig(1:numToPlot,12);
PosCEPNormal=Navsig(1:numToPlot,21);

% Angle of Attack and Drift Angle errors are next 2 columns (in mrad)
AOA=Navsig(1:numToPlot,16);
DA=Navsig(1:numToPlot,17);

% Pos errors in planes along velocity vector are next 4 columns (in fps)
PosVel=Navsig(1:numToPlot,18);
LevPosNormal=Navsig(1:numToPlot,19);
VerPosNormal=Navsig(1:numToPlot,20);

% Plot Position errors:
data = Rx(gpsLossIndex:numTimeSteps);
data = getevenodd(data,'odd');
len = length(data);
rms_x = sqrt(1/len * sum(data.^2));
rms_x = rms_x*f2nm;
display(['Final Time RMS of Pos X Error: ',num2str(rms_x)]);

data = Rx(gpsLossIndex:gpsFiveMinuteIndex);
data = getevenodd(data,'odd');
len = length(data);
rms_x = sqrt(1/len * sum(data.^2));
rms_x = rms_x*f2nm;
display(['5-minute Time RMS of Pos X Error: ',num2str(rms_x)]);

data = Ry(gpsLossIndex:numTimeSteps);
data = getevenodd(data,'odd');
len = length(data);
rms_y = sqrt(1/len * sum(data.^2));
rms_y = rms_y*f2nm;
display(['Final Time RMS of Pos Y Error: ',num2str(rms_y)]);

data = Ry(gpsLossIndex:gpsFiveMinuteIndex);
data = getevenodd(data,'odd');
len = length(data);
rms_y = sqrt(1/len * sum(data.^2));
rms_y = rms_y*f2nm;
display(['5-minute Time RMS of Pos Y Error: ',num2str(rms_y)]);

data = Rz(gpsLossIndex:numTimeSteps);
data = getevenodd(data,'odd');
len = length(data);
rms_z = sqrt(1/len * sum(data.^2));
rms_z = rms_z*f2nm;
display(['Final Time RMS of Pos Z Error: ',num2str(rms_z)]);

data = Rz(gpsLossIndex:gpsFiveMinuteIndex);
data = getevenodd(data,'odd');
len = length(data);
rms_z = sqrt(1/len * sum(data.^2));
% rms_z = rms_z*f2nm;
display(['5-minute Time RMS of Pos Z Error: ',num2str(rms_z)]);

h1=figure;
set(h1,'Name','RMS Errors: Pos, Vel, Tilt');
subplot(3,1,1), plot(Time,Rx,Time,Ry,Time,Rz);
subplot(3,1,1), title([strTitle,'   RMS Position Errors ']);
%subplot(3,1,1), xlabel(' Time (sec)');
subplot(3,1,1), ylabel('ft');
legend('X Pos Err','Y Pos Err','Z Pos Err',0);
strl(1)={['Time RMS Pos Err X after 5 min GPS loss:    ', num2str(rms_x),' nm']};
strl(2)={['Time RMS Pos Err Y after 5 min GPS loss:    ', num2str(rms_y),' nm']};
strl(3)={['Time RMS Pos Err Z after 5 min GPS loss:    ', num2str(rms_z),' ft']};
    text(floor(numToPlot/2),max(Rx),strl,'VerticalAlignment','top',...
        'HorizontalAlignment','left','FontSize',12,'FontName','Helvetica');


% Plot Velocity errors:

subplot(3,1,2), plot(Time,Vx,Time,Vy,Time,Vz);
subplot(3,1,2), title('   RMS Velocity Errors ');
%subplot(3,1,2), xlabel(' Time (sec)');
subplot(3,1,2), ylabel('fps');
legend('X Vel Err','Y Vel Err','Z Vel Err',0);

% Plot Tilt errors:

subplot(3,1,3), plot(Time,Phix,Time,Phiy,Time,Psiz);
subplot(3,1,3), title('   RMS Tilt Errors ');
subplot(3,1,3), xlabel(' Time (sec)');
subplot(3,1,3), ylabel('mrad');
legend('X Tilt','Y Tilt','Heading Error',0);

% Process other errors.

h2=figure;
set(h2,'Name','RMS Errors: Other');

% Plot 50th Percentile errors:

subplot(3,1,1), plot(Time,LevPosCEP,Time,TotPosCEP,Time,PosCEPNormal);
subplot(3,1,1), title([strTitle,'   50th Percentile Position Errors ']);
%subplot(3,1,1), xlabel(' Time (sec)');
subplot(3,1,1), ylabel('ft');
legend('Level Pos CEP','Total Pos CEP','Pos CEP in Vel Normal Plane',0);

% Plot angle errors:

subplot(3,1,2), plot(Time,AOA,Time,DA);
subplot(3,1,2), title('   Aircraft Angle Errors ');
%subplot(3,1,2), xlabel(' Time (sec)');
subplot(3,1,2), ylabel('mrad');
legend('Angle of Attack','Drift Angle',0);

% Plot Pos errors in velocity normal plane:

subplot(3,1,3), plot(Time,PosVel,Time,LevPosNormal,Time,VerPosNormal);
subplot(3,1,3), title('   RMS Pos Errors in Vel Normal Plane ');
%subplot(3,1,3), xlabel(' Time (sec)');
subplot(3,1,3), ylabel('ft');
legend('Pos along vel vector','Level pos in vel normal plane','Vert pos in vel normal plane',0);

if showandtell
    % Plot Tilt errors:
    pad=[-10;-10;-9;-9;-8;-8;-7;-7;-6;-6;-5;-5;-4;-4;-3;-3;-2;-2;-1;-1];
    Time= [pad;Time];
    Phix = [Phix(1)*ones(20,1);Phix];
    Phiy = [Phiy(1)*ones(20,1);Phiy];
    Psiz = [Psiz(1)*ones(20,1);Psiz];
    figure;
    orient landscape;
    phix_init=num2str(Phix(1),3);
    phiy_init=num2str(Phiy(1),3);
    psiz_init=num2str(Psiz(1),3);
    strl(1)={['Init X Tilt:    ' phix_init,' mrad']};
    strl(2)={['Init Y Tilt:    ' phiy_init,' mrad']};
    strl(3)={['Init Hdg Err:   ' psiz_init,'  mrad']};
    strl(4)={'GPS acquired @T=60 sec'};
    strl(5)={'GPS lost @T=1800 sec'};
    plot(Time,Phix,Time,Phiy,Time,Psiz);
    axis([-10 max(Time) 0 max(Psiz)+10]);
    text(10,Phix(1),strl,'VerticalAlignment','top',...
        'HorizontalAlignment','left','FontSize',14,'FontName','Helvetica');
    title('   RMS Tilt Errors ');
    xlabel(' Time (sec)');
    ylabel('mrad');
    legend('X Tilt','Y Tilt','Heading Error',0);
end

