function [lat] = dcm2lat(D,conv)
% This function computes the latitude and longitude of
% a point on the earth from the direction cosine matrix
% that transforms the local level coordinates to the earth
% frame.
[num_rows,num_cols] = size(D);
if num_rows ~= 3 || num_cols ~= 3
    disp(['WARNING: Number of rows or cols less than 3: ',num2str(num_rows), ' ', num2str(num_cols)]);
end;
% Check for orthogonality:
n=norm(eye(3)-D*transpose(D));
if n>.001
    disp(['WARNING: Matrix D not a DCM! Delta norm=',num2str(n)]);
end

lat = atan2(D(2,3),sqrt(1-D(2,3)^2));
if conv=='deg'
    lat = lat * 180/pi;
end
return;