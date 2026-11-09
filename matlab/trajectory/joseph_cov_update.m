function [Pplus] = joseph_cov_update(P,K,H,R)
% This function computes the Joseph update of the covariance
% matrix based on given prior P, and Kalman gain K, measurement
% matrix H and measurement noise covariance matrix R.
[m,n] = size(P);
I = eye(n);
Pplus = (I-K*H)*P*(I-K*H)'+K*R*K';

return;
