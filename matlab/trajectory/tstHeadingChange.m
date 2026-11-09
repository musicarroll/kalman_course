% tstHeadingChange.m
final_time=1000;
load tdv_test_01.mat;
SegIndex = 1;
hdg_rate = 10;
g = 9.8;
hdg(1) = 0;
Vh = 100;
cmd_roll(1)=0;
turn_accel = .5;
Segment(SegIndex,4)=turn_accel;
for i=2:final_time
	[hdg(i),cmd_roll(i)] = procHeadingChange(hdg(i-1),SegIndex,Segment,hdg_rate_limit,g, Vh);
end
figure;
minhdg = min(hdg);
maxhdg = max(hdg);
xmin=-1;
xmax=final_time+1;
ymin=minhdg-1;
ymax=maxhdg+1;

time=1:final_time;
plot(time,hdg, time, cmd_roll);
axis([xmin,xmax,ymin,ymax]);
legend('Heading','Commanded Roll');

