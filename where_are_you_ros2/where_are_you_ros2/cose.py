import numpy as np
import matplotlib.pyplot as plt
import time

# Define the system dynamics
A = np.array([[0, 1], [-1, 0]])  # Example system matrix
state = np.array([1, 0])  # Initial state
dt = 0.1  # Time step

# Initialize the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], 'bo-', markersize=5)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title("Real-Time Dynamical System Trajectory")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.ion()  # Turn on interactive mode
plt.show()

# Real-time update
x_data, y_data = [], []
for _ in range(200):
    state = state + dt * A @ state
    x_data.append(state[0])
    y_data.append(state[1])
    line.set_data(x_data, y_data)
    plt.draw()
    plt.pause(0.05)  # Simulates a control cycle

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the plot open
