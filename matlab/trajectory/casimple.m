% Simple covariance analysis simulation.
% Assumes P0, Q, and Phi already exist.
P0=Q;
Phi = expm(F); % Assume 1 Hz Kalman cycle
clear x;
clear p11;
clear k11;
num_steps=10;
for k=1:num_steps
    P1=covextrap(P0,Phi,Q);
    K=kalgain(P1,H,R);
    k11(k)=K(1,1);
    P0=covupdate(P1,K,H);
    x(k)=norm(P0-P0');
    p11(k)=P0(1,1);
end
figure;
plot(1:num_steps,x(1:num_steps));
title('Distance between P and transpose of P');
legend('Distance between P and transpose of P');
figure;
plot(1:num_steps,p11);
title('P(1,1)');
legend('P(1,1)');

figure;
plot(1:num_steps,k11);
title('K(1,1)');
legend('K(1,1)');
