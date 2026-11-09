function [vec] = skew2vec(S)
% Converts skew symmetric matrix to vector form
vec = zeros(3,1);
vec(1,1) = S(3,2);
vec(2,1) = S(1,3);
vec(3,1) = S(2,1);
return;
