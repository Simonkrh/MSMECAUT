% Single-link flexible-joint robot — Euler sim + EKF (minimal, 1 figure)
clear; clc; close all
rng(0)  % reproducible noise

% ---- Parameters (from Table 1) ----
Ks = 1.61;          % [N/m]
Jh = 0.0021;        % [kg m^2]
Jl = 0.0059;        % [kg m^2]
M  = 0.403;         % [kg]
g  = -9.81;         % [N/m]
h  = 0.06;          % [m]
Km = 0.00767;       % [N/rad/s]
Kg = 70;            % [-]
Rm = 2.6;           % [Ohm]

% ---- Simulation setup ----
T = 7; dt = 1e-3; t = 0:dt:T; N = numel(t);

% Input (simple step to excite the system)
u = zeros(1,N); 
u(t>=1) = 1;                   % 1 V after 1 s (arbitrary but OK)

% State: x = [p; q; r; s]
f = @(x,uu) [ ...
    x(3); ...
    x(4); ...
    (Ks/Jh)*x(2) - (Km^2*Kg^2)/(Rm*Jh)*x(3) + (Km*Kg)/(Rm*Jh)*uu; ...
    -(Ks/Jh + Ks/Jl)*x(2) + (Km^2*Kg^2)/(Rm*Jh)*x(3) + (M*g*h/Jl)*sin(x(1)+x(2)) ...
    - (Km*Kg)/(Rm*Jh)*uu ];

% Jacobian df/dx (for EKF)
Afun = @(x) [ ...
    0,                           0,                           1,                         0;
    0,                           0,                           0,                         1;
    0,                        Ks/Jh,        -(Km^2*Kg^2)/(Rm*Jh),                         0;
    (M*g*h/Jl)*cos(x(1)+x(2)), (M*g*h/Jl)*cos(x(1)+x(2))-(Ks/Jh+Ks/Jl), (Km^2*Kg^2)/(Rm*Jh), 0];

% ---- Truth via Euler (choose any reasonable IC) ----
X = zeros(4,N);
X(:,1) = [0.1; -0.05; 0; 0];      % small initial offsets
for k = 1:N-1
    X(:,k+1) = X(:,k) + dt * f(X(:,k), u(k));
end

% ---- Measurement: y = p + noise ----
sig_y = 5e-3;               % rad-level measurement noise
R = sig_y^2;
y = X(1,:) + sig_y*randn(1,N);
H = [1 0 0 0]; I4 = eye(4);

% ---- EKF ----
xh = zeros(4,N);            % start near but not at truth
xh(:,1) = [0; 0; 0; 0];
P  = 1e-2*eye(4);
Q  = diag([1e-8 1e-8 1e-5 1e-5]);   % small process noise (tuned)

for k = 1:N-1
    % Predict (Euler)
    xpr = xh(:,k) + dt * f(xh(:,k), u(k));
    A   = I4 + dt * Afun(xpr);        % discretized Jacobian (Euler)
    Ppr = A*P*A' + Q;

    % Update using y(k+1)
    S = H*Ppr*H' + R;
    K = (Ppr*H')/S;
    xh(:,k+1) = xpr + K*(y(k+1) - H*xpr);

    % Joseph form for numerical robustness
    P = (I4 - K*H)*Ppr*(I4 - K*H)' + K*R*K';
end

% ---- One figure: truth vs EKF (and measurement for p) ----
figure('Name','Flexible-joint robot: time series (truth vs EKF)')
subplot(4,1,1)
plot(t, y, '.', t, xh(1,:)); grid on
legend('y meas','x1 EKF')

subplot(4,1,2)
plot(t, xh(2,:)); grid on
legend('x2 EKF')

subplot(4,1,3)
plot(t, xh(3,:)); grid on
legend('x3 EKF')

subplot(4,1,4)
plot(t, xh(4,:)); grid on
legend('x4 EKF')
