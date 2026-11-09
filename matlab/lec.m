%LEC
n=30; % life of the system
r = .05; % discount rate
I = ones(n,1); % Investment costs
M = ones(n,1); % Ops & Maint. costs
F = ones(n,1); % Fuel costs
E = ones(n,1);
discount = ones(n,1);
discount(1) = 1+r;
mum = zeros(n,1);
num(1) = (I(1)+M(1)+F(1))/discount(1);
den = zeros(n,1);
den(1) = E(1)/discount(1);

for i=2:n
    I(i) = .90*I(i-1);
    M(i) = M(i-1)+.01*M(i-1); %
    F(i) = F(i-1);
    E(i) = .99* E(i-1); % aging equipment reduces energy output per year
    discount(i) = (1+r)^i;
    num(i) = (I(i)+M(i)+F(i))/discount(i);
    den(i) = E(i)/discount(i);
end

result = sum(num) / sum(den);

