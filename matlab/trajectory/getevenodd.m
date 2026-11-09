function [newdata] = getevenodd(data,evenodd)
% This function returns either all odd number or all even number elements of
% a vector. The parameter evenodd is a string either 'even' or 'odd'.
if strcmp(evenodd,'even')
    rem=0;
else
    rem=1;
end
k=1;
len = length(data);
for i=1:len
    if mod(i,2)==rem
        newdata(k)=data(i);
        k=k+1;
    end
end
return;

