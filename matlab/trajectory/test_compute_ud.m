% test_compute_ud
n=2;
I=eye(n);
avg=0;
N=100;
for k=1:N
%basic loop
    M = ran_posdefsym(n);
    [U,D] = compute_ud(M);
    Mtest = U*D*U';
    delta=norm(M-Mtest);
    disp([num2str(k),') Norm=',num2str(delta)]);
    avg = avg + delta;
    delta_vec(k)=delta;
end
avg = avg/N;
disp(['Average norm=',num2str(avg)]);
disp(['Max norm of delta=',num2str(max(delta_vec))]);
