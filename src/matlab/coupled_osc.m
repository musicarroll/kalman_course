% Coupled harmonic oscillators
% From http://fweb.wallawalla.edu/class-wiki/index.php/Coupled_Oscillator:_Coupled_Mass-Spring_System_with_Damping

Tstop = 50; % Total time run
npts = 500; % Number of points per second
mag = 0;    % This has to do with input exp. (0 = nothing, 1 = step)
m1=20;       % Mass, Spring constants, and Damping coefficients
m2=4;
k1=15;
k2=12;
b1= .1;
b2= .2;
A=[       0                 1             0           0
   -(k1/m1+k2/m1)  -(2*b1/m1+2*b2/m1)  (k2/m1)    (2*b1/m1)
          0                 0             0           1
      ( k2/m2)          (2*b2/m2)     -(k2/m2)   -(2*b2/m2)];
B=[0 0 0 0]';              % A, B, C, D are the matrices from our equations
C=[1 0 0 0
   0 1 0 0
   0 0 1 0
   0 0 0 1];
D=[0
   0
   0
   0];
X0=[5 -1 2 0]';             % The initial value matrix
sys1 = ss(A,B,C,D);        % Setting up the system for the state equations
t = linspace(0,Tstop,npts); % Setting up the time axis
u=sin(2*pi()*0.5*t);
figure(2)
%u=randn(size(t));
%u = mag*ones(size(t));      % The input
lsim(sys1,u,t,X0);         % lsim is the solver for the state equations
xlabel('time');
ylabel('postions and velocities of mass 1 and mass 2');
