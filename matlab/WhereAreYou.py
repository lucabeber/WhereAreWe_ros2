import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

# Define the initial positions of the nodes
N_0 = np.array([0, 0])
N_1 = np.array([6, 4])
N_2 = np.array([1, 2])

# Combine the nodes into a matrix
N = np.column_stack((N_0, N_1, N_2))

# Build the symmetric squared Euclidean distance matrix
D = np.zeros((N.shape[1], N.shape[1]))
for i in range(N.shape[1]):
    for j in range(N.shape[1]):
        D[i, j] = np.linalg.norm(N[:, i] - N[:, j]) ** 2

# Compute the Gram matrix
n = D.shape[0]
H = np.eye(n) - 1/n * np.ones((n, n))
G = -1/2 * H @ D @ H

# Compute the eigenvalues and eigenvectors of the Gram matrix
V, U = np.linalg.eigh(G)
rank_G = np.linalg.matrix_rank(G)

# Extract the eigenvectors corresponding to the eigenvalues different from 0
U = U[:, V > 1e-6]
V = np.diag(V[V > 1e-6])

# Compute the coordinates matrix P
P = (U @ np.sqrt(V)).T

# Plot the points contained in the matrix N and in the matrix P
plt.figure()
plt.plot(N[0, :], N[1, :], 'ro', label='Original Points')
plt.plot(P[0, :], P[1, :], 'bx', label='Estimated Points')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Original and Estimated Points')
# plt.show()

# Define the new node positions at time t and t+1
N_0_k = np.array([0, 0])
t_k = np.array([-1, 1])
N_0_k1 = N_0_k + t_k
t_k1 = np.array([-1, -1])
N_0_k2 = N_0_k1 + t_k1

# Define the new matrices N_k, N_k1, and N_k2
N_k = np.column_stack((N_0_k, N_1, N_2))
N_k1 = np.column_stack((N_0_k1, N_1, N_2))
N_k2 = np.column_stack((N_0_k2, N_1, N_2))

# Compute the distance matrices for the new matrices
def compute_distance_matrix(N):
    D = np.zeros((N.shape[1], N.shape[1]))
    for i in range(N.shape[1]):
        for j in range(N.shape[1]):
            D[i, j] = np.linalg.norm(N[:, i] - N[:, j]) ** 2
    return D

D_k = compute_distance_matrix(N_k)
D_k1 = compute_distance_matrix(N_k1)
D_k2 = compute_distance_matrix(N_k2)

# Compute the Gram matrices for the new matrices
G_k = -1/2 * H @ D_k @ H
G_k1 = -1/2 * H @ D_k1 @ H
G_k2 = -1/2 * H @ D_k2 @ H

# Compute P_k, P_k1, and P_k2
def compute_P(G):
    V, U = np.linalg.eigh(G)
    U = U[:, V > 1e-6]
    V = np.diag(V[V > 1e-6])
    return (U @ np.sqrt(V)).T

P_k = compute_P(G_k)
P_k1 = compute_P(G_k1)
P_k2 = compute_P(G_k2)

# Center P_k in p_0k
P_k -= P_k[:, [0]]
P_k1 -= P_k1[:, [0]]
P_k2 -= P_k2[:, [0]]

# Define the optimization variables
theta = 0
T = np.array([0, 0])

# Define the optimization problem
S = np.eye(2)
alpha = 1

def fun(x):
    R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                [np.sin(x[0]),  np.cos(x[0])]])
    # print("R: ", R)
    # print("x ", x[1:3])
    return np.linalg.norm(P_pre[:, 1:3] - (R @ (alpha * S @ P_cur[:, 1:3]) + x[1:3]), ord='fro')
def fun_reflected(x):
    R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                [-np.sin(x[0]),  -np.cos(x[0])]])
    return P_pre[:, 1:3] - (R @ (alpha * S @ P_cur[:, 1:3]) + x[1:3])

P_pre = P_k
P_cur = P_k1

x0 = np.array([theta, T[0], T[1]])
options = {'disp': True, 'maxiter': 100}  # Options for the optimizer
res = minimize(fun, x0, method='BFGS')
x = res.x
R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                [np.sin(x[0]),  np.cos(x[0])]])
T = x[1:3]
print("Translation Norm:", np.linalg.norm(T))

if abs(np.linalg.norm(T) - np.linalg.norm(t_k)) > 1e-4:
    S = np.diag([-1, 1])
    res = minimize(fun, x0, method='BFGS')
    x = res.x

# # Compute the rotation matrix and the translation matrix
R = np.array([[np.cos(x[0]), -np.sin(x[0])], [np.sin(x[0]), np.cos(x[0])]])
T = x[1:]
print("Translation Norm:", np.linalg.norm(T))

# Find the rotation matrix between vector T and vector t_k
u = T / np.linalg.norm(T)
v = t_k
theta = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])

# Construct the rotation matrix
R_in = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
P_k = R_in @ P_k

# Plot the results
plt.figure()
plt.plot(P_k[0, :], P_k[1, :], 'bo', markersize=10, label='P_k')
P_k1_new = R_in @ (R @ alpha * S @ P_k1 + T.reshape(-1, 1))
plt.plot(P_k1_new[0, :], P_k1_new[1, :], 'k*', markersize=10, label='P_k1_new')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Transformed Points P_{k} and P_{k1 new}')
plt.show()

# Minimize the distance between P_k1 and P_k2
def fun2(x):
    R = np.array([[np.cos(x[0]), -np.sin(x[0])], [np.sin(x[0]), np.cos(x[0])]])
    return np.linalg.norm(P_k1_new[:, 1:] - (R @ alpha * S @ P_k2[:, 1:] + x[1:].reshape(-1, 1)), 'fro')

res = minimize(fun2, x0, method='BFGS')
x = res.x
R = np.array([[np.cos(x[0]), -np.sin(x[0])], [np.sin(x[0]), np.cos(x[0])]])
T = x[1:]

if abs(np.linalg.norm(T - P_k1_new[:, 0]) - np.linalg.norm(t_k1)) > 1e-4:
    S = np.diag([-1, 1])
    res = minimize(fun2, x0, method='BFGS')
    x = res.x
    R = np.array([[np.cos(x[0]), -np.sin(x[0])], [np.sin(x[0]), np.cos(x[0])]])
    T = x[1:]

# Plot the results
plt.figure()
plt.plot(P_k[0, :], P_k[1, :], 'ro', markersize=10, label='P_k')
plt.plot(P_k1_new[0, :], P_k1_new[1, :], 'bx', markersize=10, label='P_{k1 new}')
P_k2_new = R @ alpha * S @ P_k2 + T.reshape(-1, 1)
plt.plot(P_k2_new[0, :], P_k2_new[1, :], 'g+', markersize=10, label='P_{k2 new}')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Transformed Points P_k, P_{k1 new}, and P_{k2 new}')
plt.show()

# Mirror the points with respect to the x-axis
if np.linalg.norm(P_k2_new[:, 0] - P_k1_new[:, 0] - t_k1) > 1e-6:
    u = t_k / np.linalg.norm(t_k)
    v = t_k1 / np.linalg.norm(t_k1)
    theta = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
    s = np.sign(theta)
    M = np.diag([-1, 1])
else:
    M = np.eye(2)

P_k_f = M @ P_k
P_k1_f = M @ P_k1_new
P_k2_f = M @ P_k2_new

# Plot the mirrored points
plt.figure()
plt.plot(P_k_f[0, :], P_k_f[1, :], 'ro', markersize=10, label='P_{k f}')
plt.plot(P_k1_f[0, :], P_k1_f[1, :], 'bx', markersize=10, label='P_{k1 f}')
plt.plot(P_k2_f[0, :], P_k2_f[1, :], 'g+', markersize=10, label='P_{k2 f}')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Mirrored Points P_{k f}, P_{k1 f}, and P_{k2 f}')
plt.show()

# Plot the original points
plt.figure()
plt.plot(N_k[0, :], N_k[1, :], 'ro', markersize=10, label='N_k')
plt.plot(N_k1[0, :], N_k1[1, :], 'bx', markersize=10, label='N_k1')
plt.plot(N_k2[0, :], N_k2[1, :], 'g+', markersize=10, label='N_k2')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Original Points N_k, N_k1, and N_k2')
plt.show()