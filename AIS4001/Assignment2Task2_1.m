%% Task 2.1 – Duffing EKF: Parameter estimation only (black/orange theme)
clear; clc; close all

%% True parameters (from Task 1.1)
delta = 0.05; alpha = 0; beta = 1; gam = 8; omg = 1;  % omega fixed here

%% Simulation settings
T = 10; dt = 1e-3; t = 0:dt:T; N = numel(t);

%% Duffing dynamics (truth)
f_true = @(tk,x)[ ...
    x(2); ...
    gam*cos(omg*tk) - delta*x(2) - alpha*x(1) - beta*x(1)^3 ...
];

%% Truth via RK4 (x0 = [2; 3])
X = zeros(2,N); X(:,1) = [2;3];
for k = 1:N-1
    xk = X(:,k); tk = t(k);
    k1 = f_true(tk, xk);
    k2 = f_true(tk+dt/2, xk + dt/2*k1);
    k3 = f_true(tk+dt/2, xk + dt/2*k2);
    k4 = f_true(tk+dt,   xk + dt*k3);
    X(:,k+1) = xk + dt*(k1 + 2*k2 + 2*k3 + k4)/6;
end

%% Measurements (Task 2 says all states are measurable)
% We'll measure both x and v with small noise.
sig_x = 0.02; sig_v = 0.02;
R = diag([sig_x^2, sig_v^2]);
Y = X + [sig_x*randn(1,N); sig_v*randn(1,N)];

%% EKF with augmented state z = [x; v; delta; alpha; beta; gamma]
nx = 2; np = 4; nz = nx + np;

% Initial guess (intentionally off)
z0 = [1; 4;  0.20; 0.5; 0.5; 5];   % [x0; v0; δ; α; β; γ]
zh = zeros(nz,N); zh(:,1) = z0;

% Initial covariance
P = diag([ 0.1 0.1,  0.5^2 0.5^2 0.5^2 2^2 ]);  % states, then params

% Process noise (random walk for parameters; modest for states)
q_x = 1e-6; 
q_v = 1e-4; 
q_par = [1e-8 1e-8 1e-8 1e-6]; % δ α β γ
Q = diag([q_x q_v q_par]) * dt;

% Measurement model: y = H z + noise (measure x and v)
H = [1 0 zeros(1,np); 0 1 zeros(1,np)];
I = eye(nz);

% Helpers: dynamics f_aug and Jacobian Fz (with omega fixed)
f_aug = @(tk,z) [ ...
    z(2); ...
    z(6)*cos(omg*tk) - z(3)*z(2) - z(4)*z(1) - z(5)*z(1)^3; ...
    0; 0; 0; 0 ...
];

Fz = @(tk,z) ...
[ % d[xdot]/dz
  0, 1, 0, 0, 0, 0; ...
  % d[vdot]/dz
  -(z(4)+3*z(5)*z(1)^2), -z(3), -z(2), -z(1), -z(1)^3, cos(omg*tk); ...
  % parameter rows (zeros)
  zeros(np, nz)
];

%% EKF loop (Euler prediction)
for k = 1:N-1
    tk = t(k);

    % Predict
    z_pr = zh(:,k) + dt * f_aug(tk, zh(:,k));
    A    = I + dt * Fz(tk, z_pr);
    P_pr = A*P*A' + Q;

    % Update with y(k+1)
    yk1 = Y(:,k+1);
    S = H*P_pr*H' + R;
    K = (P_pr*H')/S;
    zh(:,k+1) = z_pr + K*(yk1 - H*z_pr);

    % Joseph form
    P = (I-K*H)*P_pr*(I-K*H)' + K*R*K';
end

%% Extract final parameter estimates
dhat = zh(3,end);
ahat = zh(4,end);
bhat = zh(5,end);
ghat = zh(6,end);

%% Print true vs estimated
fprintf('\n=== Task 2.1 – Duffing EKF (parameter-only plots) ===\n');
fprintf('  delta (true = %.5f):  hat = %.5f\n', delta, dhat);
fprintf('  alpha (true = %.5f):  hat = %.5f\n', alpha, ahat);
fprintf('  beta  (true = %.5f):  hat = %.5f\n', beta,  bhat);
fprintf('  gamma (true = %.5f):  hat = %.5f\n', gam,   ghat);
fprintf('  omega (true = %.5f):  fixed (not estimated)\n', omg);

%% ==== Black background + orange styling ====
orange = [1 0.5 0];
gray   = 0.7*[1 1 1];
gridgray = 0.25*[1 1 1];

set(0,'DefaultFigureColor','k');
set(0,'DefaultAxesColor','k');
set(0,'DefaultAxesXColor','w');
set(0,'DefaultAxesYColor','w');
set(0,'DefaultAxesZColor','w');
set(0,'DefaultTextColor','w');
set(0,'DefaultLineLineWidth',1.4);

%% Plot ONLY parameter convergence
figure('Name','Parameter Convergence (Task 2.1)','Color','k');

params_true = [delta, alpha, beta, gam];
params_hat  = zh(3:6,:); % δ α β γ
labels = {'\delta','\alpha','\beta','\gamma'};

for i = 1:4
    subplot(4,1,i)
    plot(t, params_hat(i,:), 'Color', orange); hold on
    yline(params_true(i), '--', 'Color', gray);
    grid on
    ax = gca; ax.GridColor = gridgray;
    ylabel(labels{i})
    if i==1, title('Duffing EKF – Parameter Convergence'); end
    if i==4, xlabel('time (s)'); end
end
lg = legend('estimate','true','Location','best');
set(lg,'TextColor','w','EdgeColor',gridgray,'Color','k');

%% Optional: quick summary of final absolute errors
abs_err = abs(params_hat(:,end)' - params_true);
fprintf('\nFinal absolute parameter errors:\n');
fprintf('  |delta-^| = %.3e\n', abs_err(1));
fprintf('  |alpha-^| = %.3e\n', abs_err(2));
fprintf('  |beta -^| = %.3e\n', abs_err(3));
fprintf('  |gamma-^| = %.3e\n', abs_err(4));
