%% Task 2.2 – Lorenz: EKF parameter estimation only (black/orange)
clear; clc; close all
rng(0)  % reproducible

%% True parameters (Task 1.2)
sig = 10; rho = 28; bet = 8/3;

%% Simulation settings
T = 0.2; dt = 1e-3; t = 0:dt:T; N = numel(t);

%% Lorenz dynamics (truth)
f_true = @(x)[ ...
    sig*(x(2)-x(1)); ...
    x(1)*(rho - x(3)) - x(2); ...
    x(1)*x(2) - bet*x(3) ...
];

%% Truth via Euler (IC = [-8; 7; 27])
X = zeros(3,N); X(:,1) = [-8; 7; 27];
for k = 1:N-1
    X(:,k+1) = X(:,k) + dt * f_true(X(:,k));
end

%% Measurements (Task 2: all states measurable)
% y = [p;q;r] + noise
sig_y = 0.3;
R = (sig_y^2) * eye(3);
Y = X + sig_y*randn(3,N);

%% EKF with augmented state: z = [p; q; r; sigma; rho; beta]
nx = 3; np = 3; nz = nx + np;

% Initial guess (intentionally off)
z0 = [-6; 1; 20;  15; 20; 2.5];  % [p0;q0;r0;  σ; ρ; β]
zh = zeros(nz,N); zh(:,1) = z0;

% Initial covariance
P = diag([1 1 1,   10^2 10^2 5^2]);  % states small, params large

% Process noise (random walk for parameters; modest for states)
q_p = 1e-4; q_q = 1e-4; q_r = 1e-4;
q_par = [1e-6 1e-6 1e-6];           % σ ρ β
Q = diag([q_p q_q q_r q_par]) * dt;

% Measurement model: y = H z + noise (measure p,q,r)
H = [eye(3) zeros(3,np)];
I = eye(nz);

% Dynamics for augmented state
f_aug = @(z) [ ...
    z(4)*(z(2)-z(1));                 % pdot = σ(q-p)
    z(1)*(z(5) - z(3)) - z(2);        % qdot = p(ρ-r) - q
    z(1)*z(2) - z(6)*z(3);            % rdot = pq - βr
    0; 0; 0                            % parameters: random walk
];

% Jacobian wrt z = [p q r σ ρ β]
Fz = @(z) [ ...
    % rows for state derivatives
    -z(4),      z(4),     0,      (z(2)-z(1)), 0,       0;      % d f1 / dz
     (z(5)-z(3)),-1,     -z(1),   0,            z(1),   0;      % d f2 / dz
     z(2),       z(1),   -z(6),   0,            0,     -z(3);   % d f3 / dz
    % rows for parameter derivatives (zero)
    zeros(3,6)
];

%% EKF loop (Euler prediction)
for k = 1:N-1
    % Predict
    z_pr = zh(:,k) + dt * f_aug(zh(:,k));
    A    = I + dt * Fz(z_pr);
    P_pr = A*P*A' + Q;

    % Update with y(k+1)
    yk1 = Y(:,k+1);
    S = H*P_pr*H' + R;
    K = (P_pr*H')/S;
    zh(:,k+1) = z_pr + K*(yk1 - H*z_pr);

    % Joseph form for numerical robustness
    P = (I-K*H)*P_pr*(I-K*H)' + K*R*K';
end

%% Final parameter estimates
sighat = zh(4,end);
rhohat = zh(5,end);
bethat = zh(6,end);

%% Print true vs estimated
fprintf('\n=== Task 2.2 – Lorenz EKF (parameter-only plots) ===\n');
fprintf('  sigma (true = %.6f):  hat = %.6f\n', sig, sighat);
fprintf('  rho   (true = %.6f):  hat = %.6f\n', rho, rhohat);
fprintf('  beta  (true = %.6f):  hat = %.6f\n', bet, bethat);

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
figure('Name','Lorenz – Parameter Convergence (Task 2.2)','Color','k');

params_true = [sig, rho, bet];
params_hat  = zh(4:6,:);            % σ ρ β
labels = {'\sigma','\rho','\beta'};

for i = 1:3
    subplot(3,1,i)
    plot(t, params_hat(i,:), 'Color', orange); hold on
    yline(params_true(i), '--', 'Color', gray);
    grid on
    ax = gca; ax.GridColor = gridgray;
    ylabel(labels{i})
    if i==1, title('Lorenz EKF – Parameter Convergence'); end
    if i==3, xlabel('time (s)'); end
end
lg = legend('estimate','true','Location','best');
set(lg,'TextColor','w','EdgeColor',gridgray,'Color','k');

%% Optional: quick summary of final absolute errors
abs_err = abs(params_hat(:,end)' - params_true);
fprintf('\nFinal absolute parameter errors:\n');
fprintf('  |sigma-^| = %.3e\n', abs_err(1));
fprintf('  |rho  -^| = %.3e\n', abs_err(2));
fprintf('  |beta -^| = %.3e\n', abs_err(3));
