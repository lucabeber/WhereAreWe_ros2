clc 
clear all
close all

% Define the system dynamics
A = [0, 1; -1, 0];  % Example system matrix
state = [1; 0];  % Initial state
dt = 0.1;  % Time step

% Calculate distances from the target to each anchor
distances = sqrt(sum((anchors - state').^2, 2));

% Add noise to the distances
distances_noisy = distances + 0.1 * randn(n_anchor, 1);

% Initialize the recursive least squares
[H,z] = trilateration(anchors, distances_noisy);
C = eye((n_anchor-1))*0.1^2;
P = (H'*C^-1*H)^-1;
x_ls = P*H'*C^-1*z;

x_values = zeros(k, 2);
x_values(1,:) = x_ls;
P_values = P;

% Initialize the plot
figure;
hold on;
line_handle = plot(NaN, NaN, 'bo-', 'MarkerSize', 5);
plot(anchors(:,1), anchors(:,2), 'ro', 'MarkerSize', 10, 'DisplayName', 'Anchors');
line_handle_est = plot(NaN, NaN, 'g+', 'MarkerSize', 10, 'DisplayName', 'Estimated Position');
xlim([-2, 2]);
ylim([-2, 2]);
title('Real-Time Dynamical System Trajectory');
xlabel('x');
ylabel('y');
grid on;
hold off;

% Real-time update
x_data = [];
y_data = [];
for k = 2:200
    state = state + dt * A * state;
    x_data = [x_data, state(1)];
    y_data = [y_data, state(2)];

    % Calculate distances from the target to each anchor
    distances = sqrt(sum((anchors - state').^2, 2)) + 0.1 * randn(n_anchor, 1);

    % Update the recursive least squares
    [H,z] = trilateration(anchors, distances);
    [x_ls, P] = recursive_wls(x_ls, P, z, H, C);
    x_values(k,:) = x_ls;

    % Update the plot
    set(line_handle_est, 'XData', x_values(1:k,1), 'YData', x_values(1:k,2));
    set(line_handle, 'XData', x_data, 'YData', y_data);
    drawnow;
    pause(0.05);  
end



% Predict step function for Kalman filter
function [x_pred, P_pred] = predict_step(A, x, P, Q)
    x_pred = A * x;
    P_pred = A * P * A' + Q;
end

% Update step function for Kalman filter
function [x_upd, P_upd] = update_step(H, x_pred, P_pred, R, z)
    K = P_pred * H' * inv(H * P_pred * H' + R);
    x_upd = x_pred + K * (z - H * x_pred);
    P_upd = (eye(size(P_pred)) - K * H) * P_pred;
end