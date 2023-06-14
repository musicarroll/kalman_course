% plumbplot.m: A script to plot plumbbob gravity as function
% of latitude and alt
for i=1:100
  for j=1:100
    alt=100*(i-1);
    lat=90*pi/180/1000*(j-1);
    pbg(i,j)=plumbbobg(alt,lat);
  end
end

meshc(pbg);
title('Plumb-bob Gravity vs altitude and latitude');



