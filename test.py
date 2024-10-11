import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the initial points
N_0 = np.array([0, 0])
N_1 = np.array([6, 4])
N_2 = np.array([1, 2])
N = np.column_stack((N_0, N_1, N_2))

# Build the symmetric squared Euclidean distance matrix
D = np.zeros((N.shape[1], N.shape[1]))
for i in range(N.shape[1]):
    for j in range(N.shape[1]):
        D[i, j] = np.linalg.norm(N[:, i] - N[:, j])**2

# Compute the Gram matrix
n = D.shape[0]
H = np.eye(n) - (1/n) * np.ones((n, n))
G = -0.5 * H @ D @ H

# Compute the eigenvalues and eigenvectors of the Gram matrix
V, U = np.linalg.eig(G)

# Extract the eigenvectors corresponding to the eigenvalues different from 0
U = U[:, V > 1e-6]
V = np.diag(V[V > 1e-6])
P = (U @ np.sqrt(V)).T

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
            D[i, j] = np.linalg.norm(N[:, i] - N[:, j])**2
    return D

D_k = compute_distance_matrix(N_k)
D_k1 = compute_distance_matrix(N_k1)
D_k2 = compute_distance_matrix(N_k2)

# Compute the Gram matrices for the new matrices
G_k = -0.5 * H @ D_k @ H
G_k1 = -0.5 * H @ D_k1 @ H
G_k2 = -0.5 * H @ D_k2 @ H

# Compute P_k, P_k1, and P_k2
def compute_P(G):
    V, U = np.linalg.eig(G)
    U = U[:, V > 1e-6]
    V = np.diag(V[V > 1e-6])
    return (U @ np.sqrt(V)).T


# Assuming alpha and S are defined as follows:
alpha = 1
S = np.array([[1, 0], [0, 1]])

P_k = compute_P(G_k)
P_k1 = compute_P(G_k1)
P_k2 = compute_P(G_k2)

# Center P_k in p_0k
P_k -= P_k[:, 0].reshape(-1, 1)
P_k1 -= P_k1[:, 0].reshape(-1, 1)
P_k2 -= P_k2[:, 0].reshape(-1, 1)

# Define the optimization function
def fun(x):
    R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                  [np.sin(x[0]),  np.cos(x[0])]])
    return np.linalg.norm(P_k[:, 1:3] - (R @ alpha * S @ P_k1[:, 1:3] + x[1:3]), ord='fro')

# Initial guess
theta = 0
T = np.array([0, 0])

# Optimization options
options = {'disp': True, 'maxiter': 100}

# Perform the optimization
result = minimize(fun, np.array([theta, T[0], T[1]]), method='BFGS', options=options)
x = result.x

# Compute the rotation matrix and the translation matrix
R = np.array([[np.cos(x[0]), -np.sin(x[0])],
              [np.sin(x[0]),  np.cos(x[0])]])
T = x[1:3]

# Print the results
print("Rotation Matrix:\n", R)
print("Translation Vector:\n", T)
print("Translation Norm:", np.linalg.norm(T))

print("P_k: ", P_k)
print("T: ", t_k.reshape(-1, 1))
print("P_k + T : ", P_k + t_k.reshape(-1, 1))