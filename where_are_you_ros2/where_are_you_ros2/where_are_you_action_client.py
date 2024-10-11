#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav_msgs.msg import Odometry  # For reading /odom data
from uwb_interfaces.msg import UwbRange
from action_move_interfaces.action import MoveDistance  # Import custom action
import numpy as np
from scipy.optimize import minimize

def find_P(dist, D_fixed):
    n = len(dist)+1
    D = D_fixed
    # Complete D_fixied with the first row and column
    for i in range(1,n):
        D[0][i] = dist[i-1]**2
        D[i][0] = dist[i-1]**2
    
    # Compute the Gram matrix
    H = np.eye(n) - (1/n) * np.ones((n, n))
    G = -0.5 * H @ D @ H
    
    # Compute the eigenvalues and eigenvectors of the Gram matrix
    V, U = np.linalg.eig(G) 
    
    # Compute the Gram matrix
    H = np.eye(n) - 1/n * np.ones(n)
    G = -1/2 * H * D * H

    # Extract the eigenvectors corresponding to the eigenvalues different from 0
    U = U[:, V > 1e-6]
    V = np.diag(V[V > 1e-6])

    # Compute P
    P = (U @ np.sqrt(V)).T

    # Translate the points of P so that the first point is at the origin
    P = P - P[:, 0].reshape(-1, 1)

    return P

def find_roto_translation(P_pre, P_cur, t_k):
    theta = 0
    T = np.array([0,0])

    S = np.array([[1,0],[0,1]])
    alpha = 1
    
    def fun(x):
        R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                  [np.sin(x[0]),  np.cos(x[0])]])
        print("R: ", R)
        print("x ", x[1:3])
        return np.linalg.norm(P_pre[:, 1:3] - (R @ (alpha * S @ P_cur[:, 1:3]) + x[1:3]), ord='fro')
    def fun_reflected(x):
        R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                  [-np.sin(x[0]),  -np.cos(x[0])]])
        return P_pre[:, 1:3] - (R @ (alpha * S @ P_cur[:, 1:3]) + x[1:3])

    x0 = np.array([theta, T[0], T[1]])
    options = {'disp': True, 'maxiter': 100}  # Options for the optimizer
    print("test ",fun(np.array([-2.99,-0.34,-1.37])))

    # First optimization
    result = minimize(fun, x0, method='BFGS', options=options)
    x = result.x
    R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                [np.sin(x[0]),  np.cos(x[0])]])
    T = x[1:3]
    print("Translation Norm:", np.linalg.norm(T))

    if abs(np.linalg.norm(T-P_pre[:,0]) - np.linalg.norm(t_k)) > 1e-4:
        # Second optimization with reflected S
        result_reflected = minimize(fun_reflected, x0, method='BFGS', options=options)
        x = result_reflected.x
        R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                    [np.sin(x[0]),  np.cos(x[0])]])
        T = x[1:3]
        print("Translation Norm:", np.linalg.norm(T))

    
    return R, T

# class MoveDistanceClient(Node):

#     def __init__(self):
#         super().__init__('move_distance_client')

#         # Action client for the move_distance action
#         self._action_client = ActionClient(self, MoveDistance, 'move_distance')

#         # Subscribe to /odom for tracking movement
#         self.subscription = self.create_subscription(
#             Odometry,
#             '/odom',
#             self.odom_callback,
#             10
#         )

#         self.current_x = 0.0
#         self.current_y = 0.0

#         # Subscribe to /uwb for tracking movement
#         self.subscription = self.create_subscription(
#             Odometry,
#             '/odom',
#             self.odom_callback,
#             10
#         )

#     def odom_callback(self, msg):
#         """Callback to capture odometry data (current position)."""
#         self.current_x = msg.pose.pose.position.x
#         self.current_y = msg.pose.pose.position.y

#     def send_goal(self, x_goal, y_goal):
#         """Send a goal to the action server."""
#         self.get_logger().info(f'Sending goal to x: {x_goal}, y: {y_goal}')
        
#         goal_msg = MoveDistance.Goal()
#         goal_msg.x_goal = x_goal
#         goal_msg.y_goal = y_goal

#         self._action_client.wait_for_server()
#         send_goal_future = self._action_client.send_goal_async(goal_msg)
#         send_goal_future.add_done_callback(self.goal_response_callback)

#     def goal_response_callback(self, future):
#         """Handle the response from the action server."""
#         goal_handle = future.result()
        
#         if not goal_handle.accepted:
#             self.get_logger().info('Goal rejected')
#             return
        
#         self.get_logger().info('Goal accepted')

#         # Wait for the result of the goal
#         get_result_future = goal_handle.get_result_async()
#         get_result_future.add_done_callback(self.get_result_callback)

#     def get_result_callback(self, future):
#         """Handle the result from the action server."""
#         result = future.result().result
#         self.get_logger().info(f'Final position: x: {result.x_actual}, y: {result.y_actual}')
        
#         if result.x_actual == 3.0 and result.y_actual == 4.0:
#             # If the first goal is completed, send the second goal
#             self.get_logger().info('Sending second goal to move to (1, 2)')
#             self.send_goal(1.0, 2.0)
#         elif result.x_actual == 1.0 and result.y_actual == 2.0:
#             # When second goal is completed, log success
#             self.get_logger().info('Both goals successfully reached!')
#             self.perform_other_code()

#     def perform_other_code(self):
#         """Placeholder for future work after completing goals."""
#         self.get_logger().info("Performing other tasks after reaching goals...")
#         # Here you can add the code that you want to run after the goals are reached.

def main(args=None):
    rclpy.init(args=args)
    
    # client = MoveDistanceClient()

    # # Get initial odometry before sending the goal
    # client.get_logger().info(f'Initial position: x: {client.current_x}, y: {client.current_y}')
    
    
    # # Send the first goal to move to (3, 4)
    # client.send_goal(3.0, 4.0)

    # rclpy.spin(client)

    # client.destroy_node()

    # test where are we
    N_0_k = np.array([0, 0])
    t_k = np.array([-1, 1])
    N_0_k1 = N_0_k + t_k
    t_k1 = np.array([-1, -1])
    N_0_k2 = N_0_k1 + t_k1

    # Define the new matrices N_k, N_k1, and N_k2
    N_1 = np.array([6, 4])  # Example vector, replace with actual values
    N_2 = np.array([1, 2])  # Example vector, replace with actual values

    N_k = np.column_stack((N_0_k, N_1, N_2))
    N_k1 = np.column_stack((N_0_k1, N_1, N_2))
    N_k2 = np.column_stack((N_0_k2, N_1, N_2))

    # Compute the distance matrices for the new matrices
    D_k = np.zeros((N_k.shape[1], N_k.shape[1]))
    D_k1 = np.zeros((N_k1.shape[1], N_k1.shape[1]))
    D_k2 = np.zeros((N_k2.shape[1], N_k2.shape[1]))

    for i in range(N_k.shape[1]):
        for j in range(N_k.shape[1]):
            D_k[i, j] = np.linalg.norm(N_k[:, i] - N_k[:, j])**2
            D_k1[i, j] = np.linalg.norm(N_k1[:, i] - N_k1[:, j])**2
            D_k2[i, j] = np.linalg.norm(N_k2[:, i] - N_k2[:, j])**2

    # Compute the P matrices for the new matrices
    P_k = find_P(np.sqrt(D_k[0, 1:]), D_k)
    P_k1 = find_P(np.sqrt(D_k1[0, 1:]), D_k1)
    P_k2 = find_P(np.sqrt(D_k2[0, 1:]), D_k2)

    print("P_k:\n", P_k)
    print("P_k1:\n", P_k1)
    print("P_k2:\n", P_k2)

    # Compute the relative rotation and translation between the matrices
    R_k, T_k = find_roto_translation(P_k, P_k1, t_k)

    # Rototranslate the P_k1 in the new frame
    P_k1_new = R_k @ P_k1[:, 1:3] + T_k

    # Compute the relative rotation and translation between the matrices
    R_k1, T_k1 = find_roto_translation(P_k1_new, P_k2, t_k1)

    # Rototranslate the P_k2 in the new frame
    P_k2_new = R_k1 @ P_k2[:, 1:3] + T_k1

    rclpy.shutdown()


if __name__ == '__main__':
    main()

