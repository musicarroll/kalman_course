function [AttCorr] = getAttCorr(tilt_x,tilt_y,az_err_z)
% Build skew symmetric rotation matrix:
attcorrvec = [tilt_x;tilt_y;az_err_z];
attcorrmat = skew(attcorrvec);
% Subtract from identity:
A = eye(3)-attcorrmat;
% Return the inversion thereof:
AttCorr = inv(A);
return