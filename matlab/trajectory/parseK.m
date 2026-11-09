% Parse K structure
% This script assumes that K is a structure with time, i.e.,
% an output (one of three different kinds) from Simulink.
tmp=0;
myK1(1)=0;
myK2(1)=0;
myK3(1)=0;
mm=length(K.time);
for kk=1:mm
    tmp=K.signals.values(1,1,kk);
    myK1(kk)=tmp;
    tmp=K.signals.values(2,1,kk);
    myK2(kk)=tmp;
    tmp=K.signals.values(3,1,kk);
    myK3(kk)=tmp;
end
myK1=myK1';
myK2=myK2';
myK3=myK3';

h=figure;
subplot(3,1,1), plot(myK1);
subplot(3,1,2), plot(myK2);
subplot(3,1,3), plot(myK3);
