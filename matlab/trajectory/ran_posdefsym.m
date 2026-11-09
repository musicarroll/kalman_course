function [M] = ran_posdefsym(n)
% ran_posdefsym generates a random positive definite, symmetric
% matrix of order n
    ok=0;
    while ~ok 
        M=10*randn(n);
        for i=1:n
            for j=1:n
                if i>j
                    M(i,j)=M(j,i);
                end
            end
        end
        ok = isposdef(M);
    end
return;

