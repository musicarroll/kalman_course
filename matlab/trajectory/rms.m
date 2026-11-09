function rmsval = rms(vect,depvar)
% rms
% This function returns the RMS value of a time-dependent vector using numerical integration.
[m,n]=size(depvar);
last = depvar(m);
first = depvar(1);
disp(first);
disp(last);
% assume monotonically increasing
T=last-first;
sqerror=vect.*vect;
rmsval = 1/T * sqrt(integvec(sqerror,depvar));
return;

