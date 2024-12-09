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