function [K] = kalgain(P,H,R)

K = P*H'*inv(H*P*H'+R);
