% plotkalout.m -- script to plot Kalman filter outputs assumed to be
% in the workspace
figure;
k=1;
[t,s]=size(tout);
for i=1:n
    for j=1:n
        v = reshape(yout.signals(1,1).values(i,j,:),1,t);
        subplot(n,n,k); plot(v);
        k = k+1;
    end
end
