%avarstudy.m
clear;
numSamp=100;
k=0;
sigma=0.0001;
for tau=0.001:.001:10
    k=k+1;
    clock_deltas = tau + sigma*randn(1,numSamp);
    ticks = zeros(1,numSamp);
    for i=2:numSamp
        ticks(i)=ticks(i-1)+clock_deltas(i);
    end

% AVAR:

    sum=0;
    for i=1:numSamp-2
        sum = sum + (ticks(i+2)-2*ticks(i+1)+ticks(i))/tau;
    end

    avar(k) = 1/(2*(numSamp-2))* sum;
end


