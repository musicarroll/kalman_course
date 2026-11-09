function [newgain] = computeGain(oldgain)
% For recursive averaging only!!!
  newgain = oldgain / (1+oldgain);
return;
