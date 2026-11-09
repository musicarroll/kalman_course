% test_ud.m
% This program generates random matrices of various orders and tests the
% UD factorization approach to inverting them.
% Test matrix must be both symmetric and positive definite.

n=10;
I=eye(n);
avg=0;
N=100;
for k=1:N
%basic loop
    % Generate a random positive definite symmetric matrix:
    [M] = ran_posdefsym(n);
    [Minv] = ud_inv(M);
    Itest = Minv*M;
    delta=norm(I-Itest);
    disp([num2str(k),') Norm=',num2str(delta)]);
    avg = avg + delta;
    delta_vec(k)=delta;
end
avg = avg/N;
disp(['Average norm of delta=',num2str(avg)]);
disp(['Max norm of delta=',num2str(max(delta_vec))]);
