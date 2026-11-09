% Recursive Averaging

% Initial values (guess).
xhat=5; % Also used as truth.
x = 5;
gain = 1;

% Now loop for a while:
for k=1:100
  z = x + 5*randn(1);
  disp(xhat);
  xhat = xhat + gain*(z-xhat);
  err(k) = xhat-x;
  sqerr(k) = (xhat-x)^2;
  gain = computeGain(gain);
end
plot(1:100,err,1:100,sqerr);
title('Plot of Error and Square Error')
legend('xhat-x','(xhat-x)^2');
