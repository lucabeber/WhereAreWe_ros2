% clear all;
clc;
close all;
%%
N_0 = [0;0];
N_1 = [6;4];
N_2 = [1;2];
% N_3 = [2;5];
% N_4 = [3;1];

N = [N_0, N_1, N_2];

%% Build the symmetric squared Euclidean distance matrix
D = zeros(size(N,2), size(N,2));
for i = 1:size(N,2)
    for j = 1:size(N,2)
        D(i,j) = norm(N(:,i) - N(:,j))^2;
    end
end
%rank of G
%% Compute the Gram matrix
% The Gram matrix is defined as G = -1/2 * H * D * H
% where H = I - 1/n * 1 * 1^T
n = size(D,1);
H = eye(n) - 1/n * ones(n,n);
G = -1/2 * H * D * H;

%% Compute the eigenvalues and eigenvectors of the Gram matrix
[U, V] = eig(G);

% extract the eigenvectors corresponding to the eigenvalues different from 0
U = U(:,diag(V) > 1e-6);
V = V(diag(V) > 1e-6, diag(V) > 1e-6);
U 
V

P = (U * sqrt(V))'

%% Plot the points contained in the matrix N and in the matrix P
% figure;
% hold on;
% plot(N(1,:), N(2,:), 'ro');
% plot(P(1,:), P(2,:), 'bx');
% hold off;
% axis equal;
% grid on;

%% WhereAreYou 
% Given node 0 to be the reference node, so the only node that is moving. Ve define the new node position at time t and t+1
N_0_k = [0;0];
t_k = [-1;1];
N_0_k1 = N_0_k + t_k;
t_k1 = [-1;-1];
N_0_k2 = N_0_k1 + t_k1;

% Define the new matrix N_k, N_k1 and N_k2
N_k = [N_0_k, N_1, N_2];
N_k1 = [N_0_k1, N_1, N_2];
N_k2 = [N_0_k2, N_1, N_2];



% Compute the distance matrix for the new matrices
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

% Compute the Gram matrix for the new matrices
G_k = -1/2 * H * D_k * H;
G_k1 = -1/2 * H * D_k1 * H;
G_k2 = -1/2 * H * D_k2 * H;

% Compute P_k, P_k1 and P_k2
[U_k, V_k] = eig(G_k);
[U_k1, V_k1] = eig(G_k1);
[U_k2, V_k2] = eig(G_k2);

U_k = U_k(:,diag(V_k) > 1e-6);
V_k = V_k(diag(V_k) > 1e-6, diag(V_k) > 1e-6);
P_k = (U_k * sqrt(V_k))'

U_k1 = U_k1(:,diag(V_k1) > 1e-6);
V_k1 = V_k1(diag(V_k1) > 1e-6, diag(V_k1) > 1e-6);
P_k1 = (U_k1 * sqrt(V_k1))'

U_k2 = U_k2(:,diag(V_k2) > 1e-6);
V_k2 = V_k2(diag(V_k2) > 1e-6, diag(V_k2) > 1e-6);
P_k2 = (U_k2 * sqrt(V_k2))'

%% Center P_k in p_0k

P_k = P_k - P_k(:,1)
P_k1 = P_k1 - P_k1(:,1)
P_k2 = P_k2 - P_k2(:,1)

% % Plot P_k and P_k0
figure;
hold on;   
plot(P_k(1,:), P_k(2,:), 'ro');
plot(P_k0(1,:), P_k0(2,:), 'bx');
hold off;
axis equal;
grid on;
legend('P_k', 'P_k0');

%% Minimize the distance between P_k and P_k1
% We want to find the transformation matrix T that 
% arg min
% θ,T
% Pˆ = Pk − (R(θ)αSPk+1 + T),

% where R(θ) is the rotation matrix and T is the translation matrix

% Define the optimization variables
theta = 0;
T = [0;0];

% Define the optimization problem
S = [1,0;0,1];
P_k(:,2:3)
P_k1(:,2:3)
alpha = 1;
fun = @(x) norm(P_k(:,2:3) - ([cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))] * alpha * S * P_k1(:,2:3) + [x(2);x(3)]), 'fro');
x0 = [theta; T];
options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton');
[x, fval] = fminunc(fun, x0, options);

% Compute the rotation matrix and the translation matrix
R = [cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))] 
T = [x(2);x(3)]
norm(T)

if abs(norm(T) - norm(t_k)) > 1e-4
    S = [-1,0;0,1];
    fun = @(x) norm(P_k(:,2:3) - ([cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))] * S * P_k1(:,2:3) + [x(2);x(3)]), 'fro');
    x0 = [theta; T];
    options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton');
    [x, fval] = fminunc(fun, x0, options);
end 

% Compute the rotation matrix and the translation matrix
R = [cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))] 
T = [x(2);x(3)]
norm(T)

% Find the tranlation vector from the second point of P_k to the second point of P_k1
T = P_k(:,2) - P_k1(:,2);

% Translate the points of P_k1
P_k1_tmp = P_k1 + T;

% Find the rotation matrix between the second and the third point of P_k and P_k1
% Normalize the vectors
v = P_k(:,3) - P_k(:,2);
u = P_k1(:,3) - P_k1(:,2);
u = u / norm(u);
v = v / norm(v);

% Compute the angle between the vectors
theta = atan2(v(2), v(1)) - atan2(u(2), u(1));

% Construct the rotation matrix
R = [cos(theta), -sin(theta); sin(theta), cos(theta)];

% Rotate the points of P_k1 around the second point
P_k1_new = R * P_k1_tmp + (eye(2) - R) * P_k1_tmp(:,2);

% Find the translation between the first point of P_k and the first point of P_k1
T = P_k1_new(:,1) - P_k(:,1);

% Check if the translation is equal to t_k, if not mirror the points
S = [-1,0;0,1];
if norm(T - t_k) > 1e-6
    alpha = 1;
    P_k1_new = alpha * S * P_k1_new;    
    T = P_k1_new(:,1) - P_k(:,1);

    if norm(T - t_k) > 1e-6
        alpha = -1;
        P_k1_new = alpha * S * P_k1_new;    
        T = P_k1_new(:,1) - P_k(:,1);
    end
end
    

% find the rotazion matrix between vector T and vector t_k
% Normalize the vectors
u = T / norm(T);
v = t_k;

% Compute the angle between the vectors
% cos_theta = dot(u, v);
% theta = acos(cos_theta);
theta = atan2(v(2), v(1)) - atan2(u(2), u(1));

% Construct the rotation matrix
R_in = [cos(theta), -sin(theta); sin(theta), cos(theta)];
P_k = R_in * P_k;
P_k1_new = R_in * P_k1_new;

% plot the results
figure;
hold on;
% plot(N_k(1,:), N_k(2,:), 'ro','MarkerSize',10);
% plot(N_k1(1,1), N_k1(2,1), 'ro','MarkerSize',10);
% plot(N_k2(1,1), N_k2(2,1), 'ro','MarkerSize',10);
plot(P_k(1,:), P_k(2,:), 'bo','MarkerSize',10);
% plot(P_k1(1,:), P_k1(2,:), 'g+','MarkerSize',10);
% P_k1_new =  R_in *( R * alpha * S * P_k1 + T);
plot(P_k1_new(1,:), P_k1_new(2,:), 'k*','MarkerSize',10);
%plot(R * P_k1 + T, 'kx');
hold off;
axis equal;
grid on;
legend('N_k', 'N_k1', 'N_k2', 'P_k', 'P_k1', 'R * P_k1 + T');


%% Minimize the distance between P_k1 and P_k2
% Define the optimization variables
S = [1,0;0,1];
alpha = 1;
theta = 0;
T = [0;0];
P_k2t = P_k2 + P_k1_new(:,1);
fun = @(x) norm(P_k1_new(:,2:3) - ([cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))] * alpha * S * P_k2(:,2:3) + [x(2);x(3)]), 'fro');
x0 = [theta; T];
options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton');
[x, fval] = fminunc(fun, x0, options);
R = [cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))]
T = [x(2);x(3)]
norm(T-P_k1_new(:,1))

if abs(norm(T-P_k1_new(:,1)) - norm(t_k1)) > 1e-4
    S = [-1,0;0,1];
    alpha = 1;
    theta = 0;
    T = [0;0];
    P_k2t = P_k2 + P_k1_new(:,1);
    fun = @(x) norm(P_k1_new(:,2:3) - ([cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))] * alpha * S * P_k2(:,2:3) + [x(2);x(3)]), 'fro');
    x0 = [theta; T];
    options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton');
    [x, fval] = fminunc(fun, x0, options);
    R = [cos(x(1)),-sin(x(1)); sin(x(1)), cos(x(1))]
    T = [x(2);x(3)]
    norm(T-P_k1_new(:,1))
end

% plot the results
figure;
hold on;
plot(P_k(1,:), P_k(2,:), 'ro','MarkerSize',10);
plot(P_k1_new(1,:), P_k1_new(2,:), 'bx','MarkerSize',10);
P_k2_new =  R * alpha * S * P_k2 + T;
% plot(P_k2(1,:), P_k2(2,:), 'ro','MarkerSize',10);
plot(P_k2_new(1,:), P_k2_new(2,:), 'g+','MarkerSize',10);
hold off;
axis equal;
grid on;
legend('P_k', 'P_k1_new', 'P_k2_new');

%% Mirror the points with respect the x-axis
% Define the mirror matrix
if norm(P_k2_new(:,1)-P_k1_new(:,1)-t_k1) > 1e-6
    % find the angle of rotation between t_k and t_k1
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

% plot the results
figure;
hold on;
plot(P_k_f(1,:), P_k_f(2,:), 'ro','MarkerSize',10);
plot(P_k1_f(1,:), P_k1_f(2,:), 'bx','MarkerSize',10);
plot(P_k2_f(1,:), P_k2_f(2,:), 'g+','MarkerSize',10);
hold off;
axis equal;
grid on;
legend('P_k', 'P_k1', 'P_k2');

figure;
hold on;
plot(N_k(1,:), N_k(2,:), 'ro','MarkerSize',10);
plot(N_k1(1,:), N_k1(2,:), 'bx','MarkerSize',10);
plot(N_k2(1,:), N_k2(2,:), 'g+','MarkerSize',10);
hold off;
axis equal;
grid on;