function [P1] = covupdate(P0,K,H)

[m,n]= size(P0);
I=eye(m);

P1=(I-K*H)*P0;

return;