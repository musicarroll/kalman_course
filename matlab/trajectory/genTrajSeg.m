function [Segment]=genTrajSeg(n,Segment)
% This function prompts the user for information with which
% to generate a trajectory segment.
% The input is the current 2-d trajectory segment array and the index n
% of the current segment to be generated.
% The output is the input Segment augmented by one segment.
deg2rad = pi/180;
ft2m = 2.54 * 12 / 100;
g2m_per_sec_sq = 9.8;

Segment(n,1)=input(['Enter the final time for segment #',num2str(n),': ']);

Segment(n,2)=input(['Enter the time step for segment #',num2str(n),': ']);

Segment(n,3)=input(['Enter the final heading (deg) for segment #',num2str(n),': ']);
Segment(n,3)=Segment(n,3)*deg2rad;

Segment(n,4)=input(['Enter the turn (centripetal) acceleration (g) for segment #',num2str(n),': ']);
Segment(n,4)=Segment(n,4)*g2m_per_sec_sq;

Segment(n,5)=input(['Enter the final total velocity (fps) for segment #',num2str(n),': ']);
Segment(n,5)=Segment(n,5)*ft2m;

Segment(n,6)=input(['Enter the longitudinal acceleration (g) for segment #',num2str(n),': ']);
Segment(n,6)=Segment(n,6)*g2m_per_sec_sq;

Segment(n,7)=input(['Enter the final altitude (ft) for segment #',num2str(n),': ']);
Segment(n,7)=Segment(n,7)*ft2m;

Segment(n,8)=input(['Enter the altitude rate (fps) for segment #',num2str(n),': ']);
Segment(n,8)=Segment(n,8)*ft2m;

Segment(n,9)=input(['Enter the vertical acceleration (g) for segment #',num2str(n),': ']);
Segment(n,9)=Segment(n,9)*g2m_per_sec_sq;

Segment(n,10)=input(['Enter the pitch angle (climb or dive in deg) for segment #',num2str(n),': ']);
Segment(n,10)=Segment(n,10)*deg2rad;

return;
