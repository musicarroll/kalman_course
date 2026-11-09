% study Markov power spectrum
clear;
sigma=50;
beta=2400;
omega=-pi:.001:pi;
m=length(omega);
mypsd=zeros(1,m);
for i=1:m
    mypsd(i)=2*beta*sigma^2/(omega(i)^2+beta^2);
end

tau=(-beta/10):.1:(beta/10);
n=length(tau);
phi=zeros(1,n);
for i=1:n
    phi(i) = sigma^2*exp(-beta*abs(tau(i)));
end
figure;
subplot(2,1,1), plot(omega,mypsd);
legend('PSD');
title(['1st-order Markov Process with ',num2str(beta),' sec correlation time, \sigma=',num2str(sigma)]);
subplot(2,1,2), plot(tau,phi);
legend('Autocorrelation');