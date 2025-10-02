%% Task 2.3 – Flexible-joint robot: EKF parameter estimation (black/orange, params only)
clear; clc; close all
rng(0)  % reproducible

%% ---- True Parameters (Table 1) ----
Ks = 1.61;           % [N/m]
Jh = 0.0021;         % [kg m^2]
Jl = 0.0059;         % [kg m^2]
M  = 0.403;          % [kg]
g  = -9.81;          % [N/m]
h  = 0.06;           % [m]
Km = 0.00767;        % [N/rad/s]
Kg = 70;             % [-]
Rm = 2.6;            % [Ohm]

%% ---- Simulation setup ----
T = 4; dt = 1e-3; t = 0:dt:T; N = numel(t);

% Input (step to excite the system)
u = zeros(1,N); 
u(t>=1) = 1;   % 1 V after 1 s

%% ---- Dynamics (truth) ----
% x = [p; q; r; s]
f_true = @(x,uu,th) [ ...
    x(3); ...
    x(4); ...
    (th.Ks/th.Jh)*x(2) - (th.Km^2*th.Kg^2)/(th.Rm*th.Jh)*x(3) + (th.Km*th.Kg)/(th.Rm*th.Jh)*uu; ...
    -(th.Ks/th.Jh + th.Ks/th.Jl)*x(2) + (th.Km^2*th.Kg^2)/(th.Rm*th.Jh)*x(3) + (th.M*th.g*th.h/th.Jl)*sin(x(1)+x(2)) ...
     - (th.Km*th.Kg)/(th.Rm*th.Jh)*uu ...
];

theta_true = struct('Ks',Ks,'Jh',Jh,'Jl',Jl,'M',M,'g',g,'h',h,'Km',Km,'Kg',Kg,'Rm',Rm);

%% ---- Truth via Euler (IC: small offsets) ----
X = zeros(4,N);
X(:,1) = [0.1; -0.05; 0; 0];
for k = 1:N-1
    X(:,k+1) = X(:,k) + dt * f_true(X(:,k), u(k), theta_true);
end

%% ---- Measurements: all states measurable (Task 2) ----
% modest measurement noise for angles and rates
sig_p = 5e-3; sig_q = 5e-3; sig_r = 1e-2; sig_s = 1e-2;
R = diag([sig_p^2 sig_q^2 sig_r^2 sig_s^2]);
Y = X + [sig_p*randn(1,N); sig_q*randn(1,N); sig_r*randn(1,N); sig_s*randn(1,N)];

%% ---- Augmented EKF: z = [p q r s Ks Jh Jl M g h Km Kg Rm]^T ----
nx = 4; np = 9; nz = nx + np;

% Initial guess (intentionally off)
z0 = [ ...
    0; 0; 0; 0; ...             % states: p q r s
    1.4*Ks; 0.6*Jh; 1.5*Jl; ... % Ks Jh Jl
    0.7*M;  0.9*g;  1.2*h;  ... % M g h
    0.8*Km; 1.1*Kg; 1.3*Rm      % Km Kg Rm
];

zh = zeros(nz,N); zh(:,1) = z0(:);

% Initial covariance: states small, parameters large (uncertain)
P = diag([ 1e-3*ones(1,nx), ...
           (0.50*Ks)^2, (0.50*Jh)^2, (0.50*Jl)^2, ...
           (0.50*M)^2,  (0.10*abs(g))^2, (0.50*h)^2, ...
           (0.50*Km)^2, (0.20*Kg)^2, (0.50*Rm)^2 ]);

% Process noise: states modest; parameters slow random walk
Q = diag([1e-7 1e-7 1e-5 1e-5,  1e-7 1e-9 1e-9 1e-7 1e-7 1e-9 1e-9 1e-9 1e-7]) * dt;

% Measurement model: y = H z + v  (measure all 4 states)
H = [eye(4) zeros(4,np)];
I = eye(nz);

% Helpers to unpack parameters
unpack = @(z) struct('Ks',z(5),'Jh',z(6),'Jl',z(7),'M',z(8),'g',z(9), ...
                     'h',z(10),'Km',z(11),'Kg',z(12),'Rm',z(13));

% Augmented dynamics
f_aug = @(z,uu) [ ...
    z(3); ...
    z(4); ...
    (z(5)/z(6))*z(2) - ((z(11)^2*z(12)^2)/(z(13)*z(6)))*z(3) + (z(11)*z(12))/(z(13)*z(6))*uu; ...
    -(z(5)/z(6) + z(5)/z(7))*z(2) + ((z(11)^2*z(12)^2)/(z(13)*z(6)))*z(3) + (z(8)*z(9)*z(10)/z(7))*sin(z(1)+z(2)) ...
     - (z(11)*z(12))/(z(13)*z(6))*uu; ...
    zeros(np,1) ...
];

% Numerical Jacobian (central difference) for robustness
function J = numjac(fun, z, uu, hstep)
    nz_loc = numel(z);
    f0 = fun(z,uu);
    J = zeros(numel(f0), nz_loc);
    for i = 1:nz_loc
        dz = zeros(nz_loc,1); dz(i) = hstep*max(1,abs(z(i)));
        fp = fun(z+dz,uu);
        fm = fun(z-dz,uu);
        J(:,i) = (fp - fm) / (2*hstep*max(1,abs(z(i))));
    end
end

hstep = 1e-6;

%% ---- EKF loop (Euler prediction) ----
for k = 1:N-1
    % Predict
    z_pr = zh(:,k) + dt * f_aug(zh(:,k), u(k));
    A    = I + dt * numjac(f_aug, z_pr, u(k), hstep);  % discrete-time approx
    P_pr = A*P*A' + Q;

    % Update with y(k+1)
    yk1 = Y(:,k+1);
    S = H*P_pr*H' + R;
    K = (P_pr*H')/S;
    zh(:,k+1) = z_pr + K*(yk1 - H*z_pr);

    % Joseph form
    P = (I-K*H)*P_pr*(I-K*H)' + K*R*K';
end

%% ---- Final parameter estimates ----
z_end = zh(:,end);
Ks_h = z_end(5);  Jh_h = z_end(6);  Jl_h = z_end(7);
M_h  = z_end(8);  g_h  = z_end(9);  h_h  = z_end(10);
Km_h = z_end(11); Kg_h = z_end(12); Rm_h = z_end(13);

%% ---- Print true vs estimated ----
fprintf('\n=== Task 2.3 – Flexible-joint EKF (parameter-only plots) ===\n');
fprintf('  Ks  (true = %.6f):  hat = %.6f\n', Ks, Ks_h);
fprintf('  Jh  (true = %.6f):  hat = %.6f\n', Jh, Jh_h);
fprintf('  Jl  (true = %.6f):  hat = %.6f\n', Jl, Jl_h);
fprintf('  M   (true = %.6f):  hat = %.6f\n', M,  M_h);
fprintf('  g   (true = %.6f):  hat = %.6f\n', g,  g_h);
fprintf('  h   (true = %.6f):  hat = %.6f\n', h,  h_h);
fprintf('  Km  (true = %.6f):  hat = %.6f\n', Km, Km_h);
fprintf('  Kg  (true = %.6f):  hat = %.6f\n', Kg, Kg_h);
fprintf('  Rm  (true = %.6f):  hat = %.6f\n', Rm, Rm_h);

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

%% ---- Plot ONLY parameter convergence ----
figure('Name','Flexible-joint – Parameter Convergence (Task 2.3)','Color','k');

params_true = [Ks,Jh,Jl,M,g,h,Km,Kg,Rm];
params_hat  = zh(5:13,:);  % 9 params over time
labels = {'K_s','J_h','J_l','M','g','h','K_m','K_g','R_m'};

for i = 1:9
    subplot(9,1,i)
    plot(t, params_hat(i,:), 'Color', orange); hold on
    yline(params_true(i), '--', 'Color', gray);
    grid on
    ax = gca; ax.GridColor = gridgray;
    ylabel(labels{i})
    if i==1, title('Flexible-joint EKF – Parameter Convergence'); end
    if i==9, xlabel('time (s)'); end
end
lg = legend('estimate','true','Location','best');
set(lg,'TextColor','w','EdgeColor',gridgray,'Color','k');

%% ---- Optional: show quick absolute errors at end ----
abs_err = abs(params_hat(:,end)' - params_true);
fprintf('\nFinal absolute parameter errors:\n');
fprintf('  |Ks-^| = %.3e\n', abs_err(1));
fprintf('  |Jh-^| = %.3e\n', abs_err(2));
fprintf('  |Jl-^| = %.3e\n', abs_err(3));
fprintf('  |M -^| = %.3e\n', abs_err(4));
fprintf('  |g -^| = %.3e\n', abs_err(5));
fprintf('  |h -^| = %.3e\n', abs_err(6));
fprintf('  |Km-^| = %.3e\n', abs_err(7));
fprintf('  |Kg-^| = %.3e\n', abs_err(8));
fprintf('  |Rm-^| = %.3e\n', abs_err(9));
