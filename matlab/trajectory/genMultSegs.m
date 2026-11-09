function [Segment] = genMultSegs(Segment,num_segs)
% This function generates multiple trajectory segments
% by prompting the user through calls to 
% genTrajSeg.
% If there are no input arguments, then genMultSegs
% creates a new array Segment and proceeds to 
% the loop that calls genTrajSeg.  Otherwise,
% genMultSegs begins by adding a new row to 
% the given argument Segment.  In this way
% the Segment array can be saved from the workspace
% to disk and then reloaded later and more
% rows (i.e., segments) added by running this
% function.
if nargin==0
    Segment = zeros(1,10);
    next = 1;
else
    [rows,cols]=size(Segment);
    if rows==1
        next = 1;
    else
        next = rows +1;
    end
end
if nargin==1
    disp('Enter number of segments.');
    disp(['It must be greater than ',num2str(next),'.']);
    num_segs = input('Enter number of segments: ');
    % Add error handling here.
end

while (next <= num_segs)
    disp('***************************');
    disp(['Segment #',num2str(next)]);
    Segment = genTrajSeg(next,Segment);
    next = next+1;
end

return;
