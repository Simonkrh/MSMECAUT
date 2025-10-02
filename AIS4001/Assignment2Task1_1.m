clear; clc; close all
delta=.05; alpha=0; beta=1; gam=8; omg=1;
T=60; dt=1e-3; t=0:dt:T; N=numel(t); 

f  = @(tk,x)[x(2); gam*cos(omg*tk)-delta*x(2)-alpha*x(1)-beta*x(1)^3];
Fj = @(x)[0 1; -(alpha+3*beta*x(1)^2) -delta];   % ∂f/∂x

% ---- Truth via RK4 (x0=[2;3]) ----
X = zeros(2,N); 
X(:,1) = [2;3];
for k = 1:N-1
    X(:,k+1) = X(:,k) + dt * f(t(k), X(:,k));
end
% ---- Noisy position measurements ----
sig=0.05; R=sig^2; y=X(1,:)+sig*randn(1,N);
H=[1 0];

% ---- EKF (init [1;4]) ----
xh=zeros(2,N); xh(:,1)=[1;4];
P = eye(2);
Q = diag([1e-4 1e-2])*dt;
I = eye(2);

for k=1:N-1
    % predict (Euler step for the filter is fine)
    xpr = xh(:,k) + dt*f(t(k), xh(:,k));
    A   = I + dt*Fj(xpr);
    Ppr = A*P*A' + Q;

    % update (use y at k+1)
    S = H*Ppr*H' + R;
    K = (Ppr*H')/S;
    xh(:,k+1) = xpr + K*(y(k+1)-H*xpr);

    % Joseph form for numerical robustness
    P = (I-K*H)*Ppr*(I-K*H)' + K*R*K';
end

% ---- Plots ----
subplot(3,1,1)
plot(t, y, '.', t, xh(1,:)), grid on
legend('y meas','x EKF')

subplot(3,1,2)
plot(t, xh(2,:)), grid on
legend('v EKF')

% subplot(3,1,3)
% plot(X(1,:),X(2,:), xh(1,:),xh(2,:)), grid on
% legend('true','EKF')
