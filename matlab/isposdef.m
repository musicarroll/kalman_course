function [ok] = isposdef(P)
evs=eig(P);
p=length(evs);
ok=1;
for i=1:p
    if evs(p) <0
        ok=0;
    end
end
