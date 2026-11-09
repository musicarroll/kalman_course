2/3/12

from:  Michael Carroll

This zip file, kcat_files.zip, contains scripts and Simulink models to run 
kcat.mdl, the Kalman Covariance Analysis Tool, that I wrote many years ago 
while working in the INS/GPS industry.  It is a teaching tool that illustrates
the use of Matlab and Simulink to explore the covariance analysis of a 9-state
INS error Kalman filter.  

To run it, unzip everything into its own folder and add that folder to your Matlab path.
Then run genTraj.m and provide a trajectory specification (e.g., box01.mat).  Then run
state9_drv.m which will call Simulink from the command line.