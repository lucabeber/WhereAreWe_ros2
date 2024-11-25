% Recursive Least Squares for Positioning Problem
% Master and 3 Anchors

clear;
clc;

% Define the positions of the anchors (x, y)
anchors = [2, 0; 7, 1; 5, 4]; % Example positions for 3 anchors
n_anchor = size(anchors, 1);
% Define the true position of the target (not know, the position that 
% we want to estimate) 
master_true_position = [4, 2];

% Calculate distances from the target to each anchor
distances = sqrt(sum((anchors - master_true_position).^2, 2));

% What happens if we add noise to the distances?
% We add zero-mean Gaussian noise with standard deviation of 0.1 for 5 times to the distances

% Number of noisy measurements
n_mes = 1000;

% Initialize the noisy distances
distances_noisy = zeros(n_mes, n_anchor);
for i = 1:n_mes
    distances_noisy(i, :) = distances + 0.1 * randn(3, 1);
end

% Display the noisy distances
disp('Noisy distances to each anchor:');
disp(distances_noisy);

% Compute the estimated position
estimated_position = zeros(n_mes, 2);
for i = 1:n_mes
    estimated_position(i,:) = trilateration(anchors, distances_noisy(i, :));
end

% Plot the estimated positions
figure;
hold on;
plot(anchors(:,1), anchors(:,2), 'ro', 'MarkerSize', 10, 'DisplayName', 'Anchors');
plot(estimated_position(:,1), estimated_position(:,2), 'g+', 'MarkerSize', 10, 'DisplayName', 'Estimated Position');
plot(master_true_position(1), master_true_position(2), 'bx', 'MarkerSize', 10, 'DisplayName', 'True Position');
legend;
xlabel('X Position');
ylabel('Y Position');
title('2D Localization with Anchors');
grid on;
hold off;

% Let's see what happens if the standard deviation of the noise is increased to 0.5

% Initialize the noisy distances
distances_noisy_2 = zeros(n_mes, n_anchor);
for i = 1:n_mes
    distances_noisy_2(i, :) = distances + 0.5 * randn(3, 1);
end

% Compute the estimated position
estimated_position_2 = zeros(n_mes, 2);
for i = 1:n_mes
    estimated_position_2(i,:) = trilateration(anchors, distances_noisy_2(i, :));
end

% Plot the estimated positions
figure;
hold on;
plot(anchors(:,1), anchors(:,2), 'ro', 'MarkerSize', 10, 'DisplayName', 'Anchors');
plot(estimated_position_2(:,1), estimated_position_2(:,2), 'g+', 'MarkerSize', 10, 'DisplayName', 'Estimated Position');
plot(master_true_position(1), master_true_position(2), 'bx', 'MarkerSize', 10, 'DisplayName', 'True Position');
legend;
xlabel('X Position');
ylabel('Y Position');
title('2D Localization with Anchors');
grid on;
hold off;

% Now compare the estimated positions with the different noise levels
figure;
hold on;
plot(anchors(:,1), anchors(:,2), 'ro', 'MarkerSize', 10, 'DisplayName', 'Anchors');
plot(estimated_position(:,1), estimated_position(:,2), 'g+', 'MarkerSize', 10, 'DisplayName', 'Estimated Position (0.1 std)');
plot(estimated_position_2(:,1), estimated_position_2(:,2), 'b+', 'MarkerSize', 10, 'DisplayName', 'Estimated Position (0.5 std)');
plot(master_true_position(1), master_true_position(2), 'rx', 'MarkerSize', 10, 'DisplayName', 'True Position');
legend;
xlabel('X Position');
ylabel('Y Position');
title('2D Localization with Anchors');
grid on;
hold off;

% What happen is the number of anchors increases?
% Let's add a 3 anchors to the system

% Define the positions of the anchors (x, y)
anchors_2 = [2, 0; 7, 1; 5, 4; 3, 1; 1, 4; 2, -1; 3, 3]; % Example positions for 4 anchors

n_anchor_2 = size(anchors_2, 1);
% Define the true position of the target (not know, the position that 
% we want to estimate) 
master_true_position = [4, 2];

% Calculate distances from the target to each anchor
distances_2 = sqrt(sum((anchors_2 - master_true_position).^2, 2));

% What happens if we add noise to the distances?
% We add zero-mean Gaussian noise with standard deviation of 0.1 for 5 times to the distances

% Initialize the noisy distances
distances_noisy_2 = zeros(n_mes, n_anchor_2);
for i = 1:n_mes
    distances_noisy_2(i, :) = distances_2 + 0.1 * randn(n_anchor_2, 1);
end

% Display the noisy distances
% disp('Noisy distances to each anchor:');
% disp(distances_noisy);

% Compute the estimated position
estimated_position_2 = zeros(n_mes, 2);
for i = 1:n_mes
    estimated_position_2(i,:) = trilateration(anchors_2, distances_noisy_2(i, :));
end

% Plot the estimated positions
figure;
hold on;
plot(anchors_2(:,1), anchors_2(:,2), 'co', 'MarkerSize', 10, 'DisplayName', 'Anchors');
plot(estimated_position(:,1), estimated_position(:,2), 'r.', 'MarkerSize', 5, 'DisplayName', 'Estimated Position 3');
plot(estimated_position_2(:,1), estimated_position_2(:,2), 'g.', 'MarkerSize', 5, 'DisplayName', 'Estimated Position 6');
plot(master_true_position(1), master_true_position(2), 'bo', 'MarkerSize', 10, 'DisplayName', 'True Position');
legend;
xlabel('X Position');
ylabel('Y Position');
title('2D Localization with Anchors');
grid on;
hold off;

% Compare the mean square error of the estimated positions 
mse_3 = mean(sum((estimated_position - repmat(master_true_position, n_mes, 1)).^2, 2));
mse_6 = mean(sum((estimated_position_2 - repmat(master_true_position, n_mes, 1)).^2, 2));

disp('Mean Square Error for 3 anchors:');
disp(mse_3);

disp('Mean Square Error for 6 anchors:');
disp(mse_6);


% trilateration function 
function estimated_position = trilateration(anchors, distances)
    % Number of anchors
    n = size(anchors, 1);
    
    % Initialize matrices
    S = zeros(n-1, 2);
    p = zeros(n-1, 1);
    
    % Iterate over all anchors
    for i = 1:n-1
        % Fill the matrices
        S(i, :) = 2*[anchors(i+1, 1) - anchors(i, 1), anchors(i+1, 2) - anchors(i, 2)];
        p(i) = - distances(i+1)^2  + distances(i)^2 + anchors(i+1, 1)^2 - anchors(i, 1)^2 + anchors(i+1, 2)^2 - anchors(i, 2)^2;
    end

    estimated_position = (S' * S)^-1 * S' * p;
end

