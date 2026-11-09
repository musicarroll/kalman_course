% covariance_analysis.m
% Run kalgen first!!!
% close all;

N=input('Enter number of iterations: ');  % Number of iterations
display(' ');
r=input('Enter measurements growth rate (0<r<1): ');
meas_noise_growth_factor = 1.0 + r;
shortrun = 1;

%first time:
[m,n]=size(P0);
[rrows rcols] = size(R0);
Pminus  = zeros(n,n);
Pplus  = zeros(n,n);
state_gain = zeros(n,n);
pnorm1  = zeros(1,N);
pnorm2  = zeros(1,N);
knorm   = zeros(1,N);
sgnorm  = zeros(1,N);
lnorm  = zeros(1,N);
lnorm2  = zeros(1,N);
lnorm3  = zeros(1,N);
sawtooth= zeros(1,2*N);


Pplus = P0;
R = R0;
for k=1:N
    Pminus = covextrap(Pplus,Phi,Q);
    pnorm1(k)=norm(Pminus);
    L = H*Pminus*H' + R;
    lnorm(k) = norm(L);
    Linv = inv(L);
    lnorm3(k) = norm(Linv);
    L2 = eye(rrows) - Linv;
    lnorm2(k) = norm(L2);
    K = kalgain(Pminus,H,R);
    knorm(k)=norm(K);
    state_gain = eye(n)-K*H;
    sgnorm(k) = norm(state_gain);
    Pplus = joseph_cov_update(Pminus,K,H,R);
    pnorm2(k)=norm(Pplus);
    R=meas_noise_growth_factor*R;
end
sawtooth=collate(pnorm1,pnorm2);
doubletime=delta_t*collate(1:N,1:N);

figure(1);
subplot(3,1,1), plot(doubletime,sawtooth);
legend('Covariance Norm');
title('Kalman Covariance Norm Sawtooth Plot: Always scan');
subplot(3,1,2), plot(delta_t*(1:N),knorm);
legend('Kalman Gain Norm');
title('Kalman Gain Norm Plot: Always scan');
subplot(3,1,3), plot(delta_t*(1:N),sgnorm);
legend('State Gain (I-KH) Norm: Always scan');
title('State Gain Norm (I-KH) Plot: Always scan');
xlabel(['Seconds, (',num2str(1/delta_t),' Kalman cycles/sec)']);

figure(2)
subplot(3,1,1), plot(delta_t*(1:N),lnorm);
legend('L Norm');
title('L Norm Plot: Always scan');
subplot(3,1,2), plot(delta_t*(1:N),lnorm3);
legend('Norm of inv(L)');
subplot(3,1,3), plot(delta_t*(1:N),lnorm2);
legend('Norm of I-inv(L)');
xlabel(['Seconds, (',num2str(1/delta_t),' Kalman cycles/sec)']);

if shortrun ~= 1
    
% Various measurement schedules:
% scan-noscan
Pplus = P0;
R = R0;
for k=1:N
    Pminus = covextrap(Pplus,Phi,Q);
    pnorm1(k)=norm(Pminus);
    K = kalgain(Pminus,H,R);
    knorm(k)=norm(K);
    state_gain = eye(n)-K*H;
    sgnorm(k) = norm(state_gain);
    scan = mod(k,2);  % scan-noscan
    switch scan
        case 1
            Pplus = joseph_cov_update(Pminus,K,H,R);
        case 0
            Pplus=Pminus;
    end
    pnorm2(k)=norm(Pplus);
    R=meas_noise_growth_factor*R;
end    
sawtooth=collate(pnorm1,pnorm2);
doubletime=delta_t*collate(1:N,1:N);

figure(3);

subplot(3,1,1), plot(doubletime,sawtooth);
legend('Covariance Norm');
title('Kalman Covariance Norm Sawtooth Plot: scan-noscan');
subplot(3,1,2), plot(delta_t*(1:N),knorm);
legend('Kalman Gain Norm');
title('Kalman Gain Norm Plot: scan-noscan');
subplot(3,1,3), plot(delta_t*(1:N),sgnorm);
legend('State Gain (I-KH) Norm: scan-noscan');
title('State Gain Norm (I-KH) Plot: scan-noscan');
xlabel(['Seconds, (',num2str(1/delta_t),' Kalman cycles/sec)']);
% Various measurement schedules:
% scan-scan-noscan
Pplus = P0;
R = R0;
for k=1:N
    Pminus = covextrap(Pplus,Phi,Q);
    pnorm1(k)=norm(Pminus);
    K = kalgain(Pminus,H,R);
    knorm(k)=norm(K);
    state_gain = eye(n)-K*H;
    sgnorm(k) = norm(state_gain);
    scan = mod(k,3);
    switch scan
        case 1
            Pplus = joseph_cov_update(Pminus,K,H,R);
        case 2
            Pplus = joseph_cov_update(Pminus,K,H,R);
        case 0
            Pplus=Pminus;
    end
    pnorm2(k)=norm(Pplus);
    R=meas_noise_growth_factor*R;
end    
sawtooth=collate(pnorm1,pnorm2);
doubletime=delta_t*collate(1:N,1:N);

figure(4);
subplot(3,1,1), plot(doubletime,sawtooth);
legend('Covariance Norm');
title('Kalman Covariance Norm Sawtooth Plot: scan-scan-noscan');
subplot(3,1,2), plot(delta_t*(1:N),knorm);
legend('Kalman Gain Norm');
title('Kalman Gain Norm Plot: scan-scan-noscan');
subplot(3,1,3), plot(delta_t*(1:N),sgnorm);
legend('State Gain (I-KH) Norm: scan-scan-noscan');
title('State Gain Norm (I-KH) Plot: scan-scan-noscan');
xlabel(['Seconds, (',num2str(1/delta_t),' Kalman cycles/sec)']);
% Various measurement schedules:
% scan-scan-scan-noscan
Pplus = P0;
R = R0;
for k=1:N
    Pminus = covextrap(Pplus,Phi,Q);
    pnorm1(k)=norm(Pminus);
    K = kalgain(Pminus,H,R);
    knorm(k)=norm(K);
    state_gain = eye(n)-K*H;
    sgnorm(k) = norm(state_gain);
    scan = mod(k,4);
    switch scan
        case 1
            Pplus = joseph_cov_update(Pminus,K,H,R);
        case 2
            Pplus = joseph_cov_update(Pminus,K,H,R);
        case 3
            Pplus = joseph_cov_update(Pminus,K,H,R);
        case 0
            Pplus=Pminus;
    end
    pnorm2(k)=norm(Pplus);
    R=meas_noise_growth_factor*R;
end    
sawtooth=collate(pnorm1,pnorm2);

figure(5);
subplot(3,1,1), plot(doubletime,sawtooth);
legend('Covariance Norm');
title('Kalman Covariance Norm Sawtooth Plot: scan-scan-scan-noscan');
subplot(3,1,2), plot(delta_t*(1:N),knorm);
legend('Kalman Gain Norm');
title('Kalman Gain Norm Plot: scan-scan-scan-noscan');
subplot(3,1,3), plot(delta_t*(1:N),sgnorm);
legend('State Gain (I-KH) Norm');
title('State Gain Norm (I-KH) Plot: scan-scan-scan-noscan');
xlabel(['Seconds, (',num2str(1/delta_t),' Kalman cycles/sec)']);

% Various measurement schedules:
% scan-scan-noscan-noscan
Pplus = P0;
R = R0;
for k=1:N
    Pminus = covextrap(Pplus,Phi,Q);
    pnorm1(k)=norm(Pminus);
    K = kalgain(Pminus,H,R);
    knorm(k)=norm(K);
    state_gain = eye(n)-K*H;
    sgnorm(k) = norm(state_gain);
    scan = mod(k,4);
    switch scan
        case 1
            Pplus = joseph_cov_update(Pminus,K,H,R);
        case 2
            Pplus = joseph_cov_update(Pminus,K,H,R);
        case 3
            Pplus=Pminus;
        case 0
            Pplus=Pminus;
    end
    pnorm2(k)=norm(Pplus);
    R=meas_noise_growth_factor*R;
end    
sawtooth=collate(pnorm1,pnorm2);

figure(6);
subplot(3,1,1), plot(doubletime,sawtooth);
legend('Covariance Norm');
title('Kalman Covariance Norm Sawtooth Plot: scan-scan-noscan-noscan');
subplot(3,1,2), plot(delta_t*(1:N),knorm);
legend('Kalman Gain Norm');
title('Kalman Gain Norm Plot: scan-scan-noscan-noscan');
subplot(3,1,3), plot(delta_t*(1:N),sgnorm);
legend('State Gain (I-KH) Norm');
title('State Gain Norm (I-KH) Plot: scan-scan-noscan-noscan');
xlabel(['Seconds, (',num2str(1/delta_t),' Kalman cycles/sec)']);

% Various measurement schedules:
% scan-noscan-noscan-noscan
Pplus = P0;
R = R0;
for k=1:N
    Pminus = covextrap(Pplus,Phi,Q);
    pnorm1(k)=norm(Pminus);
    K = kalgain(Pminus,H,R);
    knorm(k)=norm(K);
    state_gain = eye(n)-K*H;
    sgnorm(k) = norm(state_gain);
    scan = mod(k,4);
    switch scan
        case 1
            Pplus = joseph_cov_update(Pminus,K,H,R);
        case 2
            Pplus=Pminus;
        case 3
            Pplus=Pminus;
        case 0
            Pplus=Pminus;
    end
    pnorm2(k)=norm(Pplus);
    R=meas_noise_growth_factor*R;
end    
sawtooth=collate(pnorm1,pnorm2);

figure(7);
subplot(3,1,1), plot(doubletime,sawtooth);
legend('Covariance Norm');
title('Kalman Covariance Norm Sawtooth Plot: scan-noscan-noscan-noscan');
subplot(3,1,2), plot(delta_t*(1:N),knorm);
legend('Kalman Gain Norm');
title('Kalman Gain Norm Plot: scan-noscan-noscan-noscan');
subplot(3,1,3), plot(delta_t*(1:N),sgnorm);
legend('State Gain (I-KH) Norm');
title('State Gain Norm (I-KH) Plot: scan-noscan-noscan-noscan');
xlabel(['Seconds, (',num2str(1/delta_t),' Kalman cycles/sec)']);
% Scan five times to achieve steady state, then measure
% Growth per cycle without any scans.
% 
Pplus = P0;
R = R0;
for k=1:N
    Pminus = covextrap(Pplus,Phi,Q);
    pnorm1(k)=norm(Pminus);
    K = kalgain(Pminus,H,R);
    if k>5
        Pplus=Pminus;
    else
        Pplus = joseph_cov_update(Pminus,K,H,R);
    end
        pnorm2(k)=norm(Pplus);
    R=meas_noise_growth_factor*R;
end    
sawtooth=collate(pnorm1,pnorm2);

figure(8);
plot(doubletime,sawtooth);
legend('Covariance Norm');
title('Kalman Covariance Norm Sawtooth Plot: Measurements halted after k=5');
xlabel(['Seconds, (',num2str(1/delta_t),' Kalman cycles/sec)']);

pgrowth = diff(pnorm2);
figure(8);
plot(delta_t*(1:N-1),pgrowth);
legend('Covariance Growth Rate after Measurements halted at k=5');
xlabel(['Seconds, (',num2str(1/delta_t),' Kalman cycles/sec)']);

end

