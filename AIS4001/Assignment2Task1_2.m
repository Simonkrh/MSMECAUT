% Lorenz (truth Euler) + EKF (Euler) — 1 figure minimal version
clear; clc; close all
rng(0)                               % reproducible noise

% ---- Parameters ----
sig = 10; rho = 28; bet = 8/3;
T = 40; dt = 1e-3; t = 0:dt:T; N = numel(t);

f  = @(x)[ sig*(x(2)-x(1));
           x(1)*(rho - x(3)) - x(2);
           x(1)*x(2) - bet*x(3) ];
Jf = @(x)[ -sig,      sig,      0;
           (rho-x(3)),-1,      -x(1);
            x(2),     x(1),    -bet ];

% ---- Truth via Euler (IC = (-8,7,27)) ----
X = zeros(3,N); X(:,1) = [-8; 7; 27];
for k = 1:N-1
    X(:,k+1) = X(:,k) + dt * f(X(:,k));
end

% ---- Noisy measurement: y = p + noise ----
sig_y = 0.5; R = sig_y^2;
y = X(1,:) + sig_y*randn(1,N);
H = [1 0 0]; I3 = eye(3);

% ---- EKF (IC = (-6,1,20)) ----
xh = zeros(3,N); xh(:,1) = [-6; 1; 20];
P  = 1*eye(3);
Q  = diag([1e-3 1e-3 1e-3]) * dt;

for k = 1:N-1
    % Predict
    xpr = xh(:,k) + dt*f(xh(:,k));
    A   = I3 + dt*Jf(xpr);
    Ppr = A*P*A' + Q;

    % Update
    S = H*Ppr*H' + R;
    K = (Ppr*H')/S;
    xh(:,k+1) = xpr + K*(y(k+1) - H*xpr);
    P = (I3 - K*H)*Ppr*(I3 - K*H)' + K*R*K';
end

% ---- Single Figure: time series ----
figure('Name','Lorenz + EKF (time series)')
subplot(3,1,1)
plot(t, y, '.', t, xh(1,:), 'LineWidth', 2); grid on
legend('y meas','x EKF')

subplot(3,1,2)
plot(t, xh(2,:)); grid on
legend('x2 EKF')

subplot(3,1,3)
plot(t, xh(3,:)); grid on
legend('x3 EKF')
