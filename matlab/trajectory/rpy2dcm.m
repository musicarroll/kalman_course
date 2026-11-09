function [DCM] = rpy2dcm(roll, pitch, yaw, conv)
% function [DCM] = rpy2dcm(roll, pitch, yaw) takes a 
% three scalar inputs. The first is roll, the second,
% pitch, the third yaw.  It converts these to a direction
% cosine matrix.
if strcmp(conv,'deg')
    roll  = roll * pi/180;
    pitch = pitch * pi/180;
    yaw   = yaw * pi/180;
end

sr = sin(roll);
cr = cos(roll);

sp = sin(pitch);
cp = cos(pitch);

sy = sin(yaw);
cy = cos(yaw);

DCM = [cp*cy            cp*sy           -sp;
       sr*sp*cy-cr*sy   sr*sp*sy+cr*cy  sr*cp;
       cr*sp*cy+sr*sy   cr*sp*sy-sr*cy  cr*cp ];
return;
