function [pitch] = compPitch(V, Cbn)
% procVelocityChange.m processes V changes
% as outlined in section 6.6:
Vv = getVertVelocity(V, Cbn);
Vh = sqrt(V^2-Vv^2);
pitch = atan2(Vv,Vh);

return;

    
    