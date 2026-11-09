% Test inv_u
% This function generates 100 random unit upper triangular matrices
% and feeds them to inv_u.
n=2;
I=eye(n);
avg=0;
N=100;
ok=0;
for k=1:N
%basic loop
    while ~ok 
        M=10*randn(n);
        ok = isposdef(M);
    end
    ok = 0;
    for i=1:n
        M(i,i)=1;
        for j=1:n
            if i>j
                M(i,j)=0;
            end
        end
    end
    [Minv] = inv_u(M);
    Itest = Minv*M;
    delta=norm(I-Itest);
    disp([num2str(k),') Norm=',num2str(delta)]);
    avg = avg + delta;
    delta_vec(k)=delta;
end
avg = avg/N;
disp(['Average norm of delta=',num2str(avg)]);
disp(['Max norm of delta=',num2str(max(delta_vec))]);
