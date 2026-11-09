function [Pplus] = std_cov_update(P,K,H)
% This function computes the standard update of the covariance
% matrix based on given prior P, and Kalman gain K, measurement
% matrix H.
[m,n] = size(P);
I = eye(n);
Pplus = (I-K*H)*P;

return;
