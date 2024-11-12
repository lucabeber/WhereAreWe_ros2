clear all;
clc;
close all;

% Define the initial positions of the nodes
N_0 = [0;0];
N_1 = [6;4];
N_2 = [1;2];

% Combine the nodes into a matrix
N = [N_0, N_1, N_2];

% Build the symmetric squared Euclidean distance matrix
D = zeros(size(N,2), size(N,2));
for i = 1:size(N,2)
    for j = 1:size(N,2)
        D(i,j) = norm(N(:,i) - N(:,j))^2;
    end
end

% Compute the Gram matrix
n = size(D,1);
H = eye(n) - 1/n * ones(n,n);
G = -1/2 * H * D * H;

% Compute the eigenvalues and eigenvectors of the Gram matrix
[U, V] = eig(G);
rank(G)

% Extract the eigenvectors corresponding to the eigenvalues different from 0
U = U(:,diag(V) > 1e-6);
V = V(diag(V) > 1e-6, diag(V) > 1e-6);
U 
V

% Compute the coordinates matrix P
P = (U * sqrt(V))';

% Plot the points contained in the matrix N and in the matrix P
figure;
hold on;
plot(N(1,:), N(2,:), 'ro', 'DisplayName', 'Original Points');
plot(P(1,:), P(2,:), 'bx', 'DisplayName', 'Estimated Points');
hold off;
axis equal;
grid on;
legend;
xlabel('X Position');
ylabel('Y Position');
title('Original and Estimated Points');

% Define the new node positions at time t and t+1
N_0_k = [0;0];
t_k = [-1;1];
N_0_k1 = N_0_k + t_k;
t_k1 = [-1;-1];
N_0_k2 = N_0_k1 + t_k1;

% Define the new matrices N_k, N_k1, and N_k2
N_k = [N_0_k, N_1, N_2];
N_k1 = [N_0_k1, N_1, N_2];
N_k2 = [N_0_k2, N_1, N_2];

% Compute the distance matrices for the new matrices
D_k = zeros(size(N_k,2), size(N_k,2));
D_k1 = zeros(size(N_k1,2), size(N_k1,2));
D_k2 = zeros(size(N_k2,2), size(N_k2,2));

for i = 1:size(N_k,2)
    for j = 1:size(N_k,2)
        D_k(i,j) = norm(N_k(:,i) - N_k(:,j))^2;
        D_k1(i,j) = norm(N_k1(:,i) - N_k1(:,j))^2;
        D_k2(i,j) = norm(N_k2(:,i) - N_k2(:,j))^2;
    end
end

% Compute the Gram matrices for the new matrices
G_k = -1/2 * H * D_k * H;
G_k1 = -1/2 * H * D_k1 * H;
G_k2 = -1/2 * H * D_k2 * H;

% Compute P_k, P_k1, and P_k2
[U_k, V_k] = eig(G_k);
[U_k1, V_k1] = eig(G_k1);
[U_k2, V_k2] = eig(G_k2);

U_k = U_k(:,diag(V_k) > 1e-6);
V_k = V_k(diag(V_k) > 1e-6, diag(V_k) > 1e-6);
P_k = (U_k * sqrt(V_k))';

U_k1 = U_k1(:,diag(V_k1) > 1e-6);
V_k1 = V_k1(diag(V_k1) > 1e-6, diag(V_k1) > 1e-6);
P_k1 = (U_k1 * sqrt(V_k1))';

U_k2 = U_k2(:,diag(V_k2) > 1e-6);
V_k2 = V_k2(diag(V_k2) > 1e-6, diag(V_k2) > 1e-6);
P_k2 = (U_k2 * sqrt(V_k2))';

% Center P_k in p_0k
P_k = P_k - P_k(:,1);
P_k1 = P_k1 - P_k1(:,1);
P_k2 = P_k2 - P_k2(:,1);

% Define the optimization variables
theta = 0;
T = [0;0];

% Define the optimization problem
S = [1,0;0,1];
alpha = 1;
fun = @(x) norm(P_k(:,2:3) - ([cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))] * alpha * S * P_k1(:,2:3) + [x(2);x(3)]), 'fro');
x0 = [theta; T];
options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton');
[x, ~] = fminunc(fun, x0, options);
T = [x(2);x(3)];

if abs(norm(T) - norm(t_k)) > 1e-4
    S = [-1,0;0,1];
    fun = @(x) norm(P_k(:,2:3) - ([cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))] * S * P_k1(:,2:3) + [x(2);x(3)]), 'fro');
    x0 = [theta; T];
    options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton');
    [x, ~] = fminunc(fun, x0, options);
end 

% Compute the rotation matrix and the translation matrix
R = [cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))];
T = [x(2);x(3)];
norm(T);

% Find the rotation matrix between vector T and vector t_k
u = T / norm(T);
v = t_k;
theta = atan2(v(2), v(1)) - atan2(u(2), u(1));

% Construct the rotation matrix
R_in = [cos(theta), -sin(theta); sin(theta), cos(theta)];
P_k = R_in * P_k;

% Plot the results
figure;
hold on;
plot(P_k(1,:), P_k(2,:), 'bo', 'MarkerSize', 10, 'DisplayName', 'P_k');
P_k1_new =  R_in * (R * alpha * S * P_k1 + T);
plot(P_k1_new(1,:), P_k1_new(2,:), 'k*', 'MarkerSize', 10, 'DisplayName', 'P_k1_new');
hold off;
axis equal;
grid on;
legend;
xlabel('X Position');
ylabel('Y Position');
title('Transformed Points P_{k} and P_{k1 new}');

% Minimize the distance between P_k1 and P_k2
S = [1,0;0,1];
alpha = 1;
theta = 0;
T = [0;0];
P_k2t = P_k2 + P_k1_new(:,1);
fun = @(x) norm(P_k1_new(:,2:3) - ([cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))] * alpha * S * P_k2(:,2:3) + [x(2);x(3)]), 'fro');
x0 = [theta; T];
options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton');
[x, ~] = fminunc(fun, x0, options);
R = [cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))];
T = [x(2);x(3)];
norm(T-P_k1_new(:,1));

if abs(norm(T-P_k1_new(:,1)) - norm(t_k1)) > 1e-4
    S = [-1,0;0,1];
    alpha = 1;
    theta = 0;
    T = [0;0];
    P_k2t = P_k2 + P_k1_new(:,1);
    fun = @(x) norm(P_k1_new(:,2:3) - ([cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))] * alpha * S * P_k2(:,2:3) + [x(2);x(3)]), 'fro');
    x0 = [theta; T];
    options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton');
    [x, ~] = fminunc(fun, x0, options);
    R = [cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))];
    T = [x(2);x(3)];
    norm(T-P_k1_new(:,1));
end

% Plot the results
figure;
hold on;
plot(P_k(1,:), P_k(2,:), 'ro', 'MarkerSize', 10, 'DisplayName', 'P_k');
plot(P_k1_new(1,:), P_k1_new(2,:), 'bx', 'MarkerSize', 10, 'DisplayName', 'P_{k1 new}');
P_k2_new =  R * alpha * S * P_k2 + T;
plot(P_k2_new(1,:), P_k2_new(2,:), 'g+', 'MarkerSize', 10, 'DisplayName', 'P_{k2 new}');
hold off;
axis equal;
grid on;
legend;
xlabel('X Position');
ylabel('Y Position');
title('Transformed Points P_k, P_{k1 new}, and P_{k2 new}');

% Mirror the points with respect to the x-axis
if norm(P_k2_new(:,1)-P_k1_new(:,1)-t_k1) > 1e-6
    u = t_k / norm(t_k);
    v = t_k1 / norm(t_k1);
    theta = atan2(v(2), v(1)) - atan2(u(2), u(1));
    s = sign(theta);
    M = [-1,0;0,1];
else
    M = eye(2);
end

P_k_f = M * P_k;
P_k1_f = M * P_k1_new;
P_k2_f = M * P_k2_new;

% Plot the mirrored points
figure;
hold on;
plot(P_k_f(1,:), P_k_f(2,:), 'ro', 'MarkerSize', 10, 'DisplayName', 'P_{k f}');
plot(P_k1_f(1,:), P_k1_f(2,:), 'bx', 'MarkerSize', 10, 'DisplayName', 'P_{k1 f}');
plot(P_k2_f(1,:), P_k2_f(2,:), 'g+', 'MarkerSize', 10, 'DisplayName', 'P_{k2 f}');
hold off;
axis equal;
grid on;
legend;
xlabel('X Position');
ylabel('Y Position');
title('Mirrored Points P_{k f}, P_{k1 f}, and P_{k2 f}');

% Plot the original points
figure;
hold on;
plot(N_k(1,:), N_k(2,:), 'ro', 'MarkerSize', 10, 'DisplayName', 'N_k');
plot(N_k1(1,:), N_k1(2,:), 'bx', 'MarkerSize', 10, 'DisplayName', 'N_k1');
plot(N_k2(1,:), N_k2(2,:), 'g+', 'MarkerSize', 10, 'DisplayName', 'N_k2');
hold off;
axis equal;
grid on;
legend;
xlabel('X Position');
ylabel('Y Position');
title('Original Points N_k, N_k1, and N_k2');