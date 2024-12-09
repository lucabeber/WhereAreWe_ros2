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

% Initialize the recursive least squares
[H,z,C] = trilateration(anchors, distances_noisy, 0.1);
P = (H'*inv(C)*H)^-1;
% x_ls = P*H'* inv(C) *z;
x_ls = state(1:2);


x_values = zeros(k, 3);
x_values(1,:) = [x_ls', 0];
P_values = [P, zeros(2, 1); zeros(1, 3)];
P_values(3,3) = 0.1;
P = P_values;

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
    distances = sqrt(sum((anchors - state(1:2)').^2, 2)) + 0.1 * randn(n_anchor, 1) ;
    [H,z,C] = trilateration(anchors, distances, 0.1);

    % Update step
    [x_values(k,:), P] = update_step(x_pred, P_pred, z, [H,zeros(n_anchor-1,1)], C);
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

% Update step function for Kalman filter
function [x_k_1, P_k_1] = update_step(x_k, P_k, z_k_1, H_k_1, C_new)
    K = P_k * H_k_1' * (H_k_1 * P_k * H_k_1' + C_new)^-1;
    x_k_1 = x_k + K * (z_k_1 - H_k_1 * x_k);
    P_k_1 = (eye(3) - K * H_k_1) * P_k * (eye(3) - K * H_k_1)' + K * C_new * K';
end

% trilateration function 
function [H,z,C] = trilateration(anchors, distances, noise_std)
    % Number of anchors
    n = size(anchors, 1);
    
    % Initialize matrices
    H = zeros(n-1, 2);
    z = zeros(n-1, 1);
    C = zeros(n-1);
    
    % Iterate over all anchors
    for i = 1:n-1
        % Fill the matrices
        H(i, :) = 2*[anchors(i+1, 1) - anchors(i, 1), anchors(i+1, 2) - anchors(i, 2)];
        z(i) = - distances(i+1)^2  + distances(i)^2 + anchors(i+1, 1)^2 - anchors(i, 1)^2 + anchors(i+1, 2)^2 - anchors(i, 2)^2;
        % Fill the covariance matrix
        if i == 1
            C(i,i) = 4 * noise_std^2 * (distances(i+1)^2 + distances(i)^2);
            if n > 2
                C(i,i+1) = -4 * noise_std^2 * distances(i+1)^2;
            end
        elseif i < n-1
            C(i,i-1) = -4 * noise_std^2 * distances(i)^2;
            C(i,i) = 4 * noise_std^2 * (distances(i+1)^2 + distances(i)^2);
            C(i,i+1) = -4 * noise_std^2 * distances(i+1)^2;
        else
            C(i,i-1) = -4 * noise_std^2 * distances(i)^2;
            C(i,i) = 4 * noise_std^2 * (distances(i+1)^2 + distances(i)^2);
        end
    end
end