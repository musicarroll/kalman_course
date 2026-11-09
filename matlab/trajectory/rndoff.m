function [rnd,err] = rndoff(num,numplaces,normalize)
% Round off function: rounds off num to numplaces places.
% Returns the rounded number in rnd and the error or remainder
% in err.
% If the third argument is present, this function will 
% first normalize the mantissa and decimal exp representation.
% Note: With Matlab you start getting some erroneous 
% values in err after numplaces is greater than 31 even
% for small values of x (e.g., x=.12345)
% You would get a NaN if numplaces>308, so we limit numplaces
% to that.  A warning is printed if the passed value for
% numplaces exceeds that.
savek=0;
if nargin==3     
    if normalize == 'norm';
        if (num > 1)
            start = 0;
            step = 1;
            last = 308;
        else
            start = 308;
            step = -1;
            last = 0;
        end
    else
        normnum=num;
    end
    for k=start:step:last
        tmp = num/10^k;
        if (tmp < 1)
            savek = k;
            normnum = tmp;
            break;
        end
    end
else
    normnum = num;
end
if (numplaces > 308)
    disp('Warning (rndoff): Value of number of places limited to 308');
end
mynumplaces = min(numplaces,308);
rnd = round(normnum*10^mynumplaces)/(10^mynumplaces);
err = normnum - rnd;

% Scale it back with savek:
rnd = rnd * 10^savek;
return;