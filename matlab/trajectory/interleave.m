function [newTime,newData] = interleave(time,data1,data2)
% Function interleave.m
len = min(length(data1),length(data2));
newTime  = zeros(1,2*len);
newData = zeros(1,2*len);
for k=1:len
    newTime(2*k)   = time(k);
    newTime(2*k-1) = time(k);
    newData(2*k-1) = data1(k);
    newData(2*k)   = data2(k);
end
return;
