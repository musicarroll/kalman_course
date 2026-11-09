function [ratio] = rss_ratio(a,b)
% Computes ratio of a to rss of a and b.

ratio = a / sqrt(1+a^2 + b^2);

return;
