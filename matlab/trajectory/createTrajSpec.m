% createTrajSpec.m
% Main Matlab script for Cruise
% System Trajectory Generator.
% This file causes the workspace to
% be filled with arrays that control
% the generation of the trajectory.
% After running this script, the workspace
% should be saved to a .mat file so
% that it can be used in later sessions.
% Revision History:
%  12/20/03     Added scenario description strings.

disp('Enter a scenario description as prompted below:');
author      = input('Author: ','s');
description = input('Describe the scenario (in a single line):\n','s');

% Get initial conditions:

[lat0,lon0,alt0,...
            vel0,hdg0,pitch0,roll0,...
            hdg_rate_limit,pitch_rate_limit,roll_rate_limit,...
            seg_type,num_segs...
            ] = setTrajInit;

% Cal genMultSegs
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

Segment        = zeros(1,10);  % Dummy segment to get started.
[Segment]      = genMultSegs(Segment,num_segs);       
final_comments = input('Enter any final comments after scenario generation:\n','s');        

% Prompt to save the workspace:

answer = input('Save the workspace (y/n)? ','s');
if strcmp(answer,'y')
   disp('Enter a filename (w/o the .mat extension) for this ');
   disp('trajectory specification: ');
   filename = input('Filename: ','s');
   save(filename);
   disp(['Workspace saved in file ',filename,'.mat']);
else
    disp('Workspace not saved.');
end
answer = input('Generate this trajectory? ','s');
if strcmp(answer,'y')
    disp('Now calling genTraj.m ...');
    file_loaded = 1;
    genTraj;
else
    disp('OK.  You can generate this trajectory later by executing genTraj');
    disp(['provided that you have saved the specification file']);
end
disp('createTrajSpec finished.');

