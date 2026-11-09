% inout_test_drv.m
% Tests command line interface to Simulink model
% inout_test.mdl

t=0:0.1:10;
u=sin(t);
tu = [t', u'];
[tout,x,y]=sim('inout_test',10,[],tu);

plot(t,u,tout,y);
legend('u','y');
