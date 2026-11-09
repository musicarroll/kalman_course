function [D] = compute_d(P)

% File Name:  compute_d.m
%
%
% Related Programs and Documents:  compute_ud.m
%
%
% Run-time Environment Constraints:  Must run in Matlab.
%
%
%
%    DESCRIPTION:
% This function is a wrapper around compute_ud to return
% only the U unit upper triangular matrix of the UD decomposition.


% local definitions
%	none	<or definitions if there are any>

% default settings
%	none	<or settings>

% external functions

% external string

% internal functions	<list of functions in this file>

% global data

% local data

% public functions


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Function description:
%	
%
%	Returns:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/

% private functions

[U,D] = compute_ud(P);
return;

%
%
%    $History: template.m $
% 
%

