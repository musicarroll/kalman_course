function [roll,pitch,phdg] = dcm2rpy(C)
% This function converts a direction cosine matrix to
% roll, pitch and platform heading in radians.
  roll  = atan2(C(3,2),C(3,3));
  pitch = atan2(-C(3,1),sqrt(1-C(3,1)^2));
  phdg  = atan2(C(2,1),C(1,1));
return;
