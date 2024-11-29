% Recursive Least Hquares for Positioning Problem
% Master and 3 Anchors
close all;
clear;
clc;

% Define the positions of the anchors (x, y)
anchors = [2, 0; 7, 1; 5, 4; 1, 4]; % Example positions for 3 anchors
n_anchor = size(anchors, 1);
% Define the true position of the target (not know, the position that 
% we want to estimate) 
master_true_position = [4, 2];

% Calculate distances from the target to each anchor
distances = sqrt(sum((anchors - master_true_position).^2, 2));

% What happens if we add noise to the distances?
% We add zero-mean Gaussian noise with standard deviation of 0.1 for 5 times to the distances

% Number of noisy measurements
k = 100;

% Initialize the noisy distances
distances_noisy = zeros(k, n_anchor);

for i = 1:k
    distances_noisy(i, :) = distances + 0.1 * randn(n_anchor, 1);
end

% Compute the problem matrices
H_k = zeros((n_anchor-1)*k, 2);
Z_k = zeros((n_anchor-1)*k, 1);
C_k = zeros((n_anchor-1)*k);

for i = 1:k
    l = 1 + (i-1)*(n_anchor - 1);
    [H_k(l:l+n_anchor-2,:),Z_k(l:l+n_anchor-2), C_k(l:l+n_anchor-2,l:l+n_anchor-2)] = trilateration(anchors, distances_noisy(i, :), 0.1);
end


P_k = (H_k'*C_k^-1*H_k)^-1;
x_ls = P_k*H_k'*C_k^-1*Z_k;

% Plot the estimated position
figure;
hold on;
plot(anchors(:,1), anchors(:,2), 'ro', 'MarkerSize', 10, 'DisplayName', 'Anchors');
plot(x_ls(1), x_ls(2), 'g+', 'MarkerSize', 10, 'DisplayName', 'Estimated Position');
plot(master_true_position(1), master_true_position(2), 'bx', 'MarkerSize', 10, 'DisplayName', 'True Position');
legend;
xlabel('X Position');
ylabel('Y Position');
title('2D Localization with Anchors');
grid on;
hold off;

% Recursive Least Squares
% New measurements
distances_noisy_2 = distances + 0.1 * randn(n_anchor, 1);

% x_new = x_ls + P_k*H_new'*((H_new*P_k*H_new' + 0.1^2)^-1)*(distances_noisy_2 - H_new*x_ls);
[H,z,C] = trilateration(anchors, distances_noisy_2, 0.1);
Z_new = [Z_k; z];
H_new = [H_k; H];
C_new = [C_k, zeros(size(C_k,1), size(C,2)); zeros(size(C,1), size(C_k,2)), C];
x_new = recursive_wls(x_ls, P_k, Z_new, H_new, C_new);

% Now we simulate the recursive least squares from the first measurement
% Initialize the noisy distances
distances_noisy = zeros(k, n_anchor);
std_dev = 0.1;
for i = 1:k
    distances_noisy(i, :) = distances + std_dev * randn(n_anchor, 1);
end

% Compute the problem matrices
[H,z,C] = trilateration(anchors, distances_noisy(1, :), std_dev);

P = (H'*C^-1*H)^-1;
x_ls = P*H'*C^-1*z;

% Plot the estimated position
figure;
hold on;
plot(anchors(:,1), anchors(:,2), 'ro', 'MarkerSize', 10, 'DisplayName', 'Anchors');
estimated_pos = plot(x_ls(1), x_ls(2), 'g+', 'MarkerSize', 10, 'DisplayName', 'Estimated Position');
plot(master_true_position(1), master_true_position(2), 'bx', 'MarkerSize', 10, 'DisplayName', 'True Position');
legend;
xlabel('X Position');
ylabel('Y Position');
xlim([3.8, 4.2]);
ylim([1.8, 2.2]);
title('2D Localization with Anchors with IWSL');
grid on;

x_values = zeros(k, 2);
x_values(1,:) = x_ls;
P_values = P;

for i = 2:k
    [H_k,z_k,C] = trilateration(anchors, distances_noisy(i, :), std_dev);
    [x_values(i,:),P] = recursive_wls(x_values(i-1,:)', P, z_k, H_k, C);
    % Update the plot
    set(estimated_pos, 'XData', x_values(1:i,1), 'YData', x_values(1:i,2));
    % plot(x_values(i,1), x_values(i,2), 'g+', 'MarkerSize', 10,'HandleVisibility','off');
    drawnow; 
    hold off;
    pause(0.1); 
end

% % Define the system dynamics
% A = [0, 1; -1, 0];  % Example system matrix
% state = [1; 0];  % Initial state
% dt = 0.1;  % Time step

% % Calculate distances from the target to each anchor
% distances = sqrt(sum((anchors - state').^2, 2));

% % Add noise to the distances
% distances_noisy = distances + 0.1 * randn(n_anchor, 1);

% % Initialize the recursive least squares
% [H,z,C] = trilateration(anchors, distances_noisy, 0.1);
% P = (H'*C^-1*H)^-1;
% x_ls = P*H'*C^-1*z;

% x_values = zeros(k, 2);
% x_values(1,:) = x_ls;
% P_values = P;

% % Initialize the plot
% figure;
% hold on;
% line_handle = plot(NaN, NaN, 'bo-', 'MarkerSize', 5);
% plot(anchors(:,1), anchors(:,2), 'ro', 'MarkerSize', 10, 'DisplayName', 'Anchors');
% line_handle_est = plot(NaN, NaN, 'g+', 'MarkerSize', 10, 'DisplayName', 'Estimated Position');
% xlim([-2, 2]);
% ylim([-2, 2]);
% title('Real-Time Dynamical System Trajectory');
% xlabel('x');
% ylabel('y');
% grid on;
% hold off;

% % Real-time update
% x_data = [];
% y_data = [];
% for k = 2:200
%     state = state + dt * A * state;
%     x_data = [x_data, state(1)];
%     y_data = [y_data, state(2)];

%     % Calculate distances from the target to each anchor
%     distances = sqrt(sum((anchors - state').^2, 2)) + 0.1 * randn(n_anchor, 1);

%     % Update the recursive least squares
%     [H,z,C] = trilateration(anchors, distances, 0.1);
%     [x_ls, P] = recursive_wls(x_ls, P, z, H, C);
%     x_values(k,:) = x_ls;

%     % Update the plot
%     set(line_handle_est, 'XData', x_values(1:k,1), 'YData', x_values(1:k,2));
%     set(line_handle, 'XData', x_data, 'YData', y_data);
%     drawnow;
%     pause(0.05);  
% end



% Iterative solution for the recursive least squares
function [x_k_1, P_k_1] = recursive_wls(x_k, P_k, z_k_1, H_k_1, C_new)
    S_k_1 = H_k_1*P_k*H_k_1' + C_new;   % Covariance of the residuals
    W_k_1 = P_k*H_k_1'*S_k_1^-1;        % Update gain
    x_k_1 = x_k + W_k_1*(z_k_1 - H_k_1*x_k);
    P_k_1 = (eye(size(P_k)) - W_k_1*H_k_1)*P_k;
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
            C(i,i+1) = -4 * noise_std^2 * distances(i+1)^2;
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