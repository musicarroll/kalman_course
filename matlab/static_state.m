% static_state.m
% Demonstrates the use of a recursive averaging technique 
% when the state is static:

% Truth = mu
mu=1;
sigma=0.5;
numSamples=1000;

% Measurement Samples, randomly distributed about truth:
meas = mu + sigma*randn(1,numSamples);

avg = zeros(1,numSamples);
avg(1)=meas(1);

for n=2:numSamples
    avg(n) = recursive_avg(meas(n),1/n,avg(n-1));
end

figure(1);
t=1:numSamples;
truth = mu*ones(1,numSamples);
subplot(3,1,1), plot(t,truth,t,meas,t,avg);
legend('Truth','Meas','Avg');
subplot(3,1,2), plot(t,mu-avg)
legend('Error')
subplot(3,1,3), plot(t,(mu-avg).^2);
legend('Squared Error')


