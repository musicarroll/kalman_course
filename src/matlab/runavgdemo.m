% running average demo
% Demo ideas:  1) Initial estimate under;  2) Initial estimate over;
% 3) large sigma;  4)  small sigma;  5) large sample size;  6) small sample
% size
mysigma = 0.5;
num_samples = 30;
truth = 220.6;
meas = truth + mysigma*randn(1,num_samples);
truth_vec = truth*ones(1,num_samples);
est = zeros(1,num_samples);
est(1)=210;
for k=2:num_samples
    prior_est = est(k-1);
    gain = 1/k;
    est(k) = recursive_avg(meas(k),gain,prior_est);
end
figure(1)
plot(1:num_samples,truth_vec,'b--*',1:num_samples,meas,'r--o',1:num_samples,est,'d:k');
legend('Truth','Measurements','Estimates','Location','southeast');
xlabel('Time')
ylabel('Resistance (Ohms)')
title('Simple Running Average Estimator: Resistor')
display('Error: ')
display(est(num_samples)-truth)

