% Lorenz system with RK4 integration
%  p' =  σ (q - p)
%  q' =  p (ρ - r) - q
%  r' =  p q - β r
% Parameters: σ=10, ρ=28, β=8/3
% Initial condition: (-8, 7, 27)

clear; clc;
>
% --- Parameters ---
sigma = 10;
rho   = 28;
beta  = 8/3;

% --- Initial state [p; q; r] ---
x0 = [-8; 7; 27];

% --- Time grid ---
t0 = 0;  tf = 40;  dt = 0.01;              % adjust as you like
N  = floor((tf - t0)/dt);
t  = t0 + (0:N)*dt;

% --- RK4 integration ---
X = zeros(3, N+1);  X(:,1) = x0;
for k = 1:N
    tk = t(k); xk = X(:,k);
    k1 = lorenz_rhs(tk,         xk, sigma, rho, beta);
    k2 = lorenz_rhs(tk+dt/2,    xk + dt*k1/2, sigma, rho, beta);
    k3 = lorenz_rhs(tk+dt/2,    xk + dt*k2/2, sigma, rho, beta);
    k4 = lorenz_rhs(tk+dt,      xk + dt*k3,   sigma, rho, beta);
    X(:,k+1) = xk + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end

p = X(1,:); q = X(2,:); r = X(3,:);

% --- Plots ---
figure('Name','Lorenz: 3D trajectory');
plot3(p, q, r); grid on; xlabel('p'); ylabel('q'); zlabel('r');
title('Lorenz attractor (\sigma=10,\ \rho=28,\ \beta=8/3)');

figure('Name','Time series');
plot(t, p, t, q, t, r); grid on;
xlabel('t'); ylabel('states'); legend('p','q','r'); title('Lorenz states vs. time');

figure('Name','Phase planes');
subplot(1,3,1); plot(p, q); grid on; xlabel('p'); ylabel('q'); title('q vs p');
subplot(1,3,2); plot(p, r); grid on; xlabel('p'); ylabel('r'); title('r vs p');
subplot(1,3,3); plot(q, r); grid on; xlabel('q'); ylabel('r'); title('r vs q');

% ---------- Local RHS function ----------
function dx = lorenz_rhs(~, x, sigma, rho, beta)
% x = [p; q; r]
dx = zeros(3,1);
dx(1) = sigma*(x(2) - x(1));
dx(2) = x(1)*(rho - x(3)) - x(2);
dx(3) = x(1)*x(2) - beta*x(3);
end
