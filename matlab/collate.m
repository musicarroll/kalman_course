function [collated]=collate(v1,v2)
% collate.m collates two vectors of length m into a single vector of
% length 2m.
if length(v1)==length(v2)
    m=length(v1);
    collated = zeros(1,2*m);
    j=1;
    for i=1:m
        collated(j)=v1(i);
        collated(j+1)=v2(i);
        j=j+2;
    end
else
    collated=-1;
end
return
        