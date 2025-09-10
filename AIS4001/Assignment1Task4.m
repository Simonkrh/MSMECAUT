%% Single-link flexible-joint robot — RK4 simulation (15 s)
clear; clc;

% --- Parameters (Table 1) ---
par.Ks = 1.61;          % [N/m]   spring stiffness
par.Jh = 0.0021;        % [kg m^2] inertial of hub
par.M  = 0.403;         % [kg]     link mass
par.g  = -9.81;         % [m/s^2]  gravitational constant
par.h  = 0.06;          % [m]      height of COM
par.Km = 0.00767;       % [N·m/(rad/s)] motor constant
par.Kg = 70;            % [-]      gear ratio
par.Jl = 0.0059;        % [kg m^2] load inertia
par.Rm = 2.6;           % [Ohm]    motor resistance

% --- Input u(t): change as you like ---
% Example 1 (default): zero input
u = @(t) 10;

% Example 2: 5 V step at t >= 1 s
% u = @(t) 5*(t>=1);

% --- Simulation setup ---
Tf  = 15;               % final time [s]
dt  = 1e-3;             % RK4 fixed step [s]  (adjust if needed)
t   = 0:dt:Tf;          % time vector

% States: x = [p; q; r; s]
x   = zeros(4, numel(t));
x0  = [0; 0; 0; 0];     % initial conditions (edit if needed)
x(:,1) = x0;

% --- Dynamics (Eqs. 5–8) ---
f = @(t,x) dynamics(t,x,par,u);

% --- RK4 loop ---
for k = 1:numel(t)-1
    tk = t(k);  xk = x(:,k);
    k1 = f(tk,            xk);
    k2 = f(tk + dt/2.0,   xk + dt*k1/2.0);
    k3 = f(tk + dt/2.0,   xk + dt*k2/2.0);
    k4 = f(tk + dt,       xk + dt*k3);
    x(:,k+1) = xk + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4);
end

% --- Plots ---
figure; 
subplot(2,1,1);
plot(t, x(1,:), t, x(2,:)); grid on;
xlabel('Time [s]'); ylabel('Angle [rad]');
legend('p(t) (hub)','q(t) (link)','Location','best');
title('Flexible-joint robot: angles');

subplot(2,1,2);
plot(t, x(3,:), t, x(4,:)); grid on;
xlabel('Time [s]'); ylabel('Angular rate [rad/s]');
legend('r(t) = \dot p','s(t) = \dot q','Location','best');
title('Angular rates');

%% ----------------- Local dynamics function -----------------
function dx = dynamics(t,x,par,u)
% x = [p; q; r; s]
p = x(1); q = x(2); r = x(3); s = x(4);

Ks = par.Ks; Jh = par.Jh; Jl = par.Jl;
Km = par.Km; Kg = par.Kg; Rm = par.Rm;
M  = par.M;  g  = par.g;  h  = par.h;

% Equations (5)–(8)
pdot = r;
qdot = s;

rdot =  (Ks/Jh)*q ...
      - (Km^2*Kg^2)/(Rm*Jh)*r ...
      + (Km*Kg)/(Rm*Jh)*u(t);

sdot = -(Ks/Jh + Ks/Jl)*q ...
      + (Km^2*Kg^2)/(Rm*Jh)*r ...
      + (M*g*h/Jl)*sin(p + q) ...
      - (Km*Kg)/(Rm*Jh)*u(t);

dx = [pdot; qdot; rdot; sdot];
end
