function [U] = compute_u(P)
%
% File Name:  compute_u.m
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
%

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

[U,D] = compute_ud(P);
return;


% private functions

%
%
%    $History: compute_u.m $
%
