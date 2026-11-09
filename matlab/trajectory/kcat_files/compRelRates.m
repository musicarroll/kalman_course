function [rho_n, rho_e, rho_d, alpha_dot, Rlat, Rlon] = compRelRates(V, Cbn, lat, alt, seg_type)
% compRelRates.m computes relative rates in north-east-down coordinates.
% as outlined in section 6.7:
% Change Log:
% 5/11/04 Cbn per T. De Vries suggestion:
% 5/18/04:  Forced rates to 0 if they are small enough.

Vned = Cbn*[V;0;0];
Vn = Vned(1);
Ve = Vned(2);

[Rlat, Rlon, ae, D] = getRadii(lat, alt);
% Trying to flip sign.  Was:
%rho_n = Ve/Rlon;
%rho_e = -Vn/Rlat;
% Is (4/7/04 -- discrepancy between cmigits.pdf and Cruise Spec:
% Commented out on 5/11/04 due to Cbn transpose change above.
%rho_n = -Ve/Rlon;
%rho_e = Vn/Rlat;
% Back to what it was:
rho_n = Ve/Rlon;
if abs(rho_n) < eps
  rho_n = 0;
end

rho_e = -Vn/Rlat;
if abs(rho_e) < eps
  rho_e = 0;
end

% 0=great circle; 1=rhumb line
if seg_type ==1  % rhumb line
	rho_d = - rho_n * tan(lat);
	alpha_dot = 0;
else    % great circle
	rho_d = 0;
	alpha_dot = rho_n*tan(lat);
end

return;



