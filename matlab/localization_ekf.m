clc 
clear all
close all

dT = 1/30;

% Define the system dynamics
fun = @(x, y, theta, vel, omega) [x + vel * cos(theta) * dT; y + vel * sin(theta) * dT; theta + omega * dT]; 

% Define the system matrix
A = @(x, y, theta, vel, omega) [1, 0, -vel * sin(theta) * dT; 0, 1, vel * cos(theta) * dT; 0, 0, dT];

G = zeros(3, 3);

nu = [0;0;0];

Q = 0.1 * eye(3);

% Define the initial state
state = [4; 2; 0];  % Initial state

k = 300;

% Define anchors
anchors = [2, 0; 7, 1; 1, 1; 0, 5]; % Example positions for 3 anchors
n_anchor = size(anchors, 1);

% Calculate distances from the target to each anchor
distances = sqrt(sum((anchors - state(1:2)').^2, 2));

% Add noise to the distances
distances_noisy = distances + 0.1 * randn(n_anchor, 1);

% Initialize problem matrices
x_i = [1;1;0];
P = 1 * eye(3);


x_values = zeros(k, 3);
x_values(1,:) = [x_i'];

% Initialize the plot
figure;
hold on;
line_handle = plot(NaN, NaN, 'bo-', 'MarkerSize', 5, 'DisplayName', 'Real Trajectory');
plot(anchors(:,1), anchors(:,2), 'ro', 'MarkerSize', 10, 'DisplayName', 'Anchors');
line_handle_est = plot(NaN, NaN, 'g+', 'MarkerSize', 10, 'DisplayName', 'Estimated Position');
title('Real-Time Dynamical System Trajectory');
xlabel('x');
ylabel('y');
grid on;
legend;

% Real-time update
x_data = [];
y_data = [];
vel = 1;
omega = 0.4;
trace_P = zeros(200,1);
trace_P(1) = trace(P);

for k = 2:200
    % Simulate the system
    state = fun(state(1), state(2), state(3), vel, omega);

    % Predict step
    [x_pred, P_pred] = predict_step([x_values(k-1,1), x_values(k-1,2), x_values(k-1,3), vel, omega], P, A, G, Q, fun);

    % Calculate distances from the target to each anchor
    z = sqrt(sum((anchors - state(1:2)').^2, 2)) + 0.1 * randn(n_anchor, 1) ;

    % Compute observation jacobian
    H = observation_jacobian_H(anchors, x_pred(1:2));

        
    % Measurement model
    h = sqrt(sum((anchors - x_pred(1:2)').^2, 2));

    % Update step
    [x_values(k,:), P] = update_step(x_pred, P_pred, z, h, H, 0.1 * eye(n_anchor));
    x_data = [x_data, state(1)];
    y_data = [y_data, state(2)];

    % Update the plot
    set(line_handle_est, 'XData', x_values(1:k,1), 'YData', x_values(1:k,2));
    set(line_handle, 'XData', x_data, 'YData', y_data);
    drawnow;
    pause(0.1);  

    trace_P(k) = trace(P);
end

% Plot the results of the estimation
figure;
plot(trace_P)
title('Trace of the covariance matrix');
xlabel('Time step');
ylabel('Trace of the covariance matrix');


% Predict step function for Kalman filter
function [x_pred, P_pred] = predict_step(x, P, A, G, Q, fun)
    x_pred = fun(x(1), x(2), x(3), x(4), x(5)); % Assuming x contains [x, y, theta, vel, omega]
    A_c = A(x(1), x(2), x(3), x(4), x(5));
    P_pred = A_c * P * A_c' + G * Q * G';
end

% Iterative solution for the recursive least squares
function [x_k_1, P_k_1] = update_step(x_k, P_k, z_k_1, h_k_1, H_k_1, C_new)
    S = (H_k_1 * P_k * H_k_1' + C_new);
    K = P_k * H_k_1' * S^-1;
    x_k_1 = x_k + K * (z_k_1 - h_k_1);
    P_k_1 = (eye(3) - K * H_k_1) * P_k * (eye(3) - K * H_k_1)' + K * C_new * K';
end

% trilateration function 
function H = observation_jacobian_H(anchors, master)
    % Number of anchors
    n = size(anchors, 1);
    
    % Initialize matrices
    H = zeros(n, 3);

    % Iterate over all anchors
    for i = 1:n
        % Fill the matrices
        H(i, :) = [
    -(anchors(i, 1) - master(1)) / sqrt((anchors(i, 1) - master(1))^2 + (anchors(i, 2) - master(2))^2), ...
    -(anchors(i, 2) - master(2)) / sqrt((anchors(i, 1) - master(1))^2 + (anchors(i, 2) - master(2))^2), 0
    ];
    end
end