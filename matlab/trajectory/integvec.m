function integral = integvec(vect,t)
% Assumes a row vector for both vect and t
[m,n]=size(t);
s=0;
for i=1:m-1
 %   disp(s);
    s=s+vect(i+1)*(t(i+1)-t(i));
end
integral = s;
return;