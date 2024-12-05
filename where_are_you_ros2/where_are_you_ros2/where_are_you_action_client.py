#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav_msgs.msg import Odometry  # For reading /odom data
from sensor_msgs.msg import Range
from action_move_interfaces.action import MoveDistance  # Import custom action
from action_move_interfaces.action import Rotate
from action_move_interfaces.msg import LimoLocalization
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import tf_transformations
import os



# class KalmanFilter():
#     def __init__(self, robot, anchors, distances, noise_std):
#         self.anchors = anchors
#         self.noise_std = noise_std

#         self.H = np.zeros((3,1))
#         self.C = np.zeros(1)
#         self.z = np.zeros(1)
#         self.trilateration(anchors, distances, noise_std)

#         self.P = np.linalg.inv(self.H.T @ np.linalg.inv(self.C) @ self.H)
#         self.x_prev = np.array([robot[0], robot[1], 0])

#         self.P_values = np.zeros((3, 3))
#         self.P_values[:2, :2] = self.P
#         self.P_values[2, 2] = 0.1

        



#     def trilateration(self, anchors, distances, noise_std):
#         n = anchors.shape[0]
#         H = np.zeros((n-1, 2))
#         z = np.zeros(n-1)
#         C = np.zeros((n-1, n-1))
#         for i in range(n-1):
#             H[i, :] = 2 * (anchors[i+1, :] - anchors[i, :])
#             z[i] = -distances[i+1]**2 + distances[i]**2 + np.sum(anchors[i+1, :]**2) - np.sum(anchors[i, :]**2)
#             if i == 0:
#                 C[i, i] = 4 * noise_std**2 * (distances[i+1]**2 + distances[i]**2)
#                 if n > 2:
#                     C[i, i+1] = -4 * noise_std**2 * distances[i+1]**2
#             elif i < n-2:
#                 C[i, i-1] = -4 * noise_std**2 * distances[i]**2
#                 C[i, i] = 4 * noise_std**2 * (distances[i+1]**2 + distances[i]**2)
#                 C[i, i+1] = -4 * noise_std**2 * distances[i+1]**2
#             else:
#                 C[i, i-1] = -4 * noise_std**2 * distances[i]**2
#                 C[i, i] = 4 * noise_std**2 * (distances[i+1]**2 + distances[i]**2)
        
#         self.H[1:2] = H
#         self.z = z
#         self.C = C

#     def predict_step(self, G, Q):
#         self.fun(self.x_prev[0], self.x_prev[1], self.x_prev[2], self.x_prev[3], self.x_prev[4])
#         x = self.x_prev
#         P = self.P_values
#         x_pred = self.fun(x[0], x[1], x[2], x[3], x[4])
#         A_c = self.A(x[0], x[1], x[2], x[3], x[4])
#         P_pred = A_c @ P @ A_c.T + G @ Q @ G.T
#         return x_pred, P_pred

#     def update_step(self, x_k, P_k, z_k_1, H_k_1, C_new):
#         K = P_k @ H_k_1.T @ np.linalg.inv(H_k_1 @ P_k @ H_k_1.T + C_new)
#         x_k_1 = x_k + K @ (z_k_1 - H_k_1 @ x_k)
#         P_k_1 = (np.eye(3) - K @ H_k_1) @ P_k @ (np.eye(3) - K @ H_k_1).T + K @ C_new @ K.T
#         return x_k_1, P_k_1
    
#     def fun(self, x, y, theta, vel, omega, dT):
#         return np.array([x + vel * np.cos(theta) * dT, 
#                          y + vel * np.sin(theta) * dT, 
#                          theta + omega * dT])
    
#     def A(self, x, y, theta, vel, omega, dT):
#         return np.array([
#             [1, 0, -vel * np.sin(theta) * dT],
#             [0, 1, vel * np.cos(theta) * dT],
#             [0, 0, 1]
#         ])

class WhereAreYou():
    def __init__(self):
        
        self.anchor_dist = 2.124
        # Initialize the matrix of distances between the anchors
        self.D_fixed = np.array([[.0, .0, .0],
                                [.0, .0, self.anchor_dist],
                                [.0, self.anchor_dist, .0]])
        

        # Initialize the estimated points P_k, P_k1, and P_k2
        self.P_k = np.zeros((2, 3))
        self.P_k1 = np.zeros((2, 3))
        self.P_k2 = np.zeros((2, 3))

        self.P_relative = np.zeros((2, 3))


    def where_are_you(self, dist_1, dist_2, dist_3, t_k, t_k1):
        # Compute the P matrices for the new matrices
        P_k = self.find_P(dist_1, self.D_fixed)
        P_k1 = self.find_P(dist_2, self.D_fixed)
        P_k2 = self.find_P(dist_3, self.D_fixed)

        print("P_k:\n", P_k)
        print("P_k1:\n", P_k1)
        print("P_k2:\n", P_k2)

        # Compute the relative rotation and translation between the matrices
        R_k, T_k = self.find_roto_translation(P_k, P_k1, t_k)
        # Find the rotation matrix between vector T and vector t_k
        u = T_k / np.linalg.norm(T_k)
        v = np.array([1, 0])
        theta = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])

        # Construct the rotation matrix
        R_in = np.array([[np.cos(theta[0]), -np.sin(theta[0])],
                        [np.sin(theta[0]),  np.cos(theta[0])]])   
        P_k = R_in @ P_k

        # Rototranslate the P_k1 in the new frame
        P_k1_new = R_in @ (R_k @ P_k1 + T_k)
        print("P_k1_new:\n", P_k1_new)
        # # Compute the relative rotation and translation between the matrices
        R_k1, T_k1 = self.find_roto_translation(P_k1_new, P_k2, t_k1+t_k)

        # Rototranslate the P_k2 in the new frame
        P_k2_new = R_k1 @ P_k2 + T_k1

        # Mirror 
        if np.linalg.norm(P_k2_new[:, 0] - P_k1_new[:, 0] + t_k1)  > np.linalg.norm(P_k1_new[:, 0] + t_k1)*0.05:
            u = t_k / np.linalg.norm(t_k)
            v = t_k1 / np.linalg.norm(t_k1)
            theta = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
            s = np.sign(theta)
            M = np.array([[-1, 0], [0, 1]])
        else:
            M = np.eye(2)  
        M = np.eye(2)
        self.P_k_f = M @ P_k
        self.P_k1_f = M @ P_k1_new
        self.P_k2_f = M @ P_k2_new 

        return self.P_k_f, self.P_k1_f, self.P_k2_f
        
    
    # Find the P matrix that represents the points in the new frame
    def find_P(self, dist, D_fixed):
        n = len(dist)+1
        print("dist:", dist)    
        D = np.copy(D_fixed)
        # Complete D_fixied with the first row and column
        for i in range(1,n):
            D[0][i] = dist[i-1]**2
            D[i][0] = dist[i-1]**2
        print("D:\n", D)
        # Compute the Gram matrix
        H = np.eye(n) - (1/n) * np.ones((n, n))
        G = -0.5 * H @ D @ H
        
        # Compute the eigenvalues and eigenvectors of the Gram matrix
        V, U = np.linalg.eig(G) 
        
        # Extract the eigenvectors corresponding to the eigenvalues different from 0
        U = U[:, V > 1e-6]
        V = np.diag(V[V > 1e-6])

        # Compute P
        P = (U @ np.sqrt(V)).T

        # Translate the points of P so that the first point is at the origin
        P = P - P[:, 0].reshape(-1, 1)

        return P

    # Find the relative rotation and translation between two matrices
    def find_roto_translation(self, P_pre, P_cur, t_k):
        theta = 0
        T = np.array([0,0])

        alpha = 1
        
        def fun(x):
            R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                    [np.sin(x[0]),  np.cos(x[0])]])
            return np.linalg.norm(P_pre[:, 1:3] - (R @ (alpha * S @ P_cur[:, 1:3]) + np.array([[x[1]],[x[2]]])), ord='fro')

        x0 = np.array([theta, T[0], T[1]])
        # Decrease minimum step size to increase precision
        options = {'disp': True, 'maxiter': 100}  # Options for the optimizer

        S = np.eye(2)
        # First optimization using 
        result = minimize(fun, x0, method='BFGS', options=options)
        x = result.x
        R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                    [np.sin(x[0]),  np.cos(x[0])]]) @ S
        T = np.array([[x[1]],[x[2]]])
        print("Translation Norm:", np.linalg.norm(T))
        print("Translation: t_k:", np.linalg.norm(t_k))
        
        if abs(np.linalg.norm(T) - np.linalg.norm(t_k)) > np.linalg.norm(t_k)*0.05:
            # Second optimization with reflected S
            S = np.diag([-1, 1])
            result_reflected = minimize(fun, x0, method='BFGS', options=options)
            x = result_reflected.x
            R = np.array([[np.cos(x[0]), -np.sin(x[0])],
                        [np.sin(x[0]),  np.cos(x[0])]]) @ S
            T = np.array([[x[1]],[x[2]]])
            print("Translation Norm:", np.linalg.norm(T))

        
        return R, T

class MoveDistanceClient(Node):

    def __init__(self):
        super().__init__('move_distance_client')

        # Action client for the move_distance action
        self._action_client = ActionClient(self, MoveDistance, 'move_distance')
        self._action_client_rotate = ActionClient(self, Rotate, 'rotate')

        # Subscribe to /odom for tracking movement
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Initialize publisher for the relative localization
        self.localization_publisher = self.create_publisher(LimoLocalization, '/limo_localization', 10)

        # Initialize current position and angle
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_angle = 0.0

        # Initialize current distance between the anchors
        self.current_dist_1 = 0.0
        self.current_dist_2 = 0.0

        


        # Subscribe to /uwb for tracking movement
        self.subscription2 = self.create_subscription(
            Range,
            '/uwb/range',
            self.uwb_callback,
            10
        )
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.status = 0
        self.flag = True
        
        # Parameters for the relative localization
        self.ancor_dist = 2

        self.where_are_you = WhereAreYou()

        self.t_k = np.array([0.0,1.0])
        self.t_k1 = np.array([-1.0,0.0])

        # Angle between the vectors t_k and t_k1
        u = self.t_k / np.linalg.norm(self.t_k)
        v = self.t_k1 / np.linalg.norm(self.t_k1)
        self.theta = (np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])) * 180.0 / np.pi
        print("Theta:", self.theta)

        


        

    def timer_callback(self):
        
        if self.status == 0:
            if self.flag:
                self.dist_1 = np.array([self.current_dist_1, self.current_dist_2])
                self.get_logger().info(f'The initial distance between the anchors is: {self.dist_1[0]}, {self.dist_1[1]}')
                self.send_goal(self.t_k[0], self.t_k[1])
                self.flag = False
        if self.status == 1:
            if self.flag:
                self.send_rotation(self.theta)
                self.flag = False
        if self.status == 2:
            if self.flag:
                self.dist_2 = np.array([self.current_dist_1, self.current_dist_2])
                self.get_logger().info(f'The distance between the anchors after the first movement is: {self.dist_2[0]}, {self.dist_2[1]}')
                self.send_goal(self.t_k1[0], self.t_k1[1])
                self.flag = False
                # os.system('ros2 run ')
        if self.status == 3:
            self.get_logger().info('Both goals successfully reached! Starting relative localization')
            ## relative localization
            if self.flag:
                self.dist_3 = np.array([self.current_dist_1, self.current_dist_2])
                self.flag = False
                self.get_logger().info(f'The distance between the anchors after the second movement is: {self.dist_3[0]}, {self.dist_3[1]}')

                P_k, P_k1, P_k2 = self.where_are_you.where_are_you(self.dist_1, self.dist_2, self.dist_3, self.t_k, self.t_k1) 
                # Find the relative position between the anchors and the robot using where are you
                
                # Plot the results
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(P_k[0, :], P_k[1, :], 'bo', markersize=10, label='P_k')
                ax.plot(P_k1[0, :], P_k1[1, :], 'k*', markersize=10, label='P_k1')
                ax.plot(P_k2[0, :], P_k2[1, :], 'rs', markersize=10, label='P_k2')
                ax.axis('equal')
                ax.legend()
                ax.grid(True)
                plt.show()


                # # Publish relative localization
                # msg = LimoLocalization()
                # msg.anchor1_x = self.P_k_f[0, 1]
                # msg.anchor1_y = self.P_k_f[1, 1]
                # msg.anchor2_x = self.P_k_f[0, 2]
                # msg.anchor2_y = self.P_k_f[1, 2]
                # msg.robot_x = self.P_k_f[0, 0]
                # msg.robot_y = self.P_k_f[1, 0]
                # self.localization_publisher.publish(msg)

                # msg.robot_x = self.P_k1_f[0, 0]
                # msg.robot_y = self.P_k1_f[1, 0]
                # self.localization_publisher.publish(msg)

                # msg.robot_x = self.P_k2_f[0, 0]
                # msg.robot_y = self.P_k2_f[1, 0]
                # self.localization_publisher.publish(msg)
                # Launch node to control the robot with the keyboard
                os.system('ros2 run teleop_twist_keyboard teleop_twist_keyboard')

            
        
    def odom_callback(self, msg):
        """Callback to capture odometry data (current position)."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = tf_transformations.euler_from_quaternion(orientation_list)
        self.current_angle = yaw

    def uwb_callback(self, msg):
        """Callback to capture uwb data (current distance between the anchors)."""
        if msg.header.frame_id == 'uwb1':
            self.current_dist_1 = msg.range
        elif msg.header.frame_id == 'uwb2':
            self.current_dist_2 = msg.range

    def send_goal(self, x_goal, y_goal):
        """Send a goal to the action server."""
        self.get_logger().info(f'Sending goal to x: {x_goal}, y: {y_goal}')
        
        goal_msg = MoveDistance.Goal()
        goal_msg.x = x_goal
        goal_msg.y = y_goal

        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle the response from the action server."""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        
        self.get_logger().info('Goal accepted')

        # Wait for the result of the goal
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        self.get_logger().info(f'Final position: x: {result.actual_x}, y: {result.actual_y}')

        # Save 
        
        
        self.status += 1
        self.flag = True
    
    def send_rotation(self, angle):
        """Send a goal to the action server."""
        self.get_logger().info(f'Sending rotation goal to {angle} degrees')
        
        goal_msg = Rotate.Goal()
        goal_msg.rotation_angle = angle

        self._action_client_rotate.wait_for_server()
        send_goal_future = self._action_client_rotate.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback_rotation)

    def goal_response_callback_rotation(self, future):
        """Handle the response from the action server."""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        
        self.get_logger().info('Goal accepted')

        # Wait for the result of the goal
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback_rotation)
    
    def get_result_callback_rotation(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        self.get_logger().info(f'Final angle: {result.actual_rotation_angle}')
        
        self.status += 1
        self.flag = True

    

def main(args=None):
    rclpy.init(args=args)
    client = MoveDistanceClient()

    # Get initial odometry before sending the goal
    client.get_logger().info(f'Initial position: x: {client.current_x}, y: {client.current_y}')

    rclpy.spin(client)

    client.destroy_node()

    # # test where are we
    # where_are_you = WhereAreYou()

    # # # Test the where are you function
    # dist_1 = np.array([1,np.sqrt(2)])
    # dist_2 = np.array([np.sqrt(2),1])
    # dist_3 = np.array([np.sqrt(5),2])
    # t_k = np.array([0.0,1.0])
    # t_k1 = np.array([-1.0,0.0])

    # # init the plot
    # # where_are_you.init_plot()

    # # Call the where are you function
    # where_are_you.where_are_you(dist_1, dist_2, dist_3, t_k, t_k1) 
    
    # # Compute the distance matrices for the new matrices
    # D_k = np.zeros((N_k.shape[1], N_k.shape[1]))
    # D_k1 = np.zeros((N_k1.shape[1], N_k1.shape[1]))
    # D_k2 = np.zeros((N_k2.shape[1], N_k2.shape[1]))

    # for i in range(N_k.shape[1]):
    #     for j in range(N_k.shape[1]):
    #         D_k[i, j] = np.linalg.norm(N_k[:, i] - N_k[:, j])**2
    #         D_k1[i, j] = np.linalg.norm(N_k1[:, i] - N_k1[:, j])**2
    #         D_k2[i, j] = np.linalg.norm(N_k2[:, i] - N_k2[:, j])**2

    # # Compute the P matrices for the new matrices
    # P_k = find_P(np.sqrt(D_k[0, 1:]), D_k)
    # P_k1 = find_P(np.sqrt(D_k1[0, 1:]), D_k1)
    # P_k2 = find_P(np.sqrt(D_k2[0, 1:]), D_k2)

    # print("P_k:\n", P_k)
    # print("P_k1:\n", P_k1)
    # print("P_k2:\n", P_k2)

    # # Compute the relative rotation and translation between the matrices
    # R_k, T_k = find_roto_translation(P_k, P_k1, t_k)
    # # Find the rotation matrix between vector T and vector t_k
    # u = T_k / np.linalg.norm(T_k)
    # v = np.array([1, 0])
    # theta = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])

    # # Construct the rotation matrix
    # R_in = np.array([[np.cos(theta[0]), -np.sin(theta[0])],
    #                 [np.sin(theta[0]),  np.cos(theta[0])]])   
    # P_k = R_in @ P_k

    # # Rototranslate the P_k1 in the new frame
    # P_k1_new = R_in @ (R_k @ P_k1 + T_k)

    # # # Compute the relative rotation and translation between the matrices
    # R_k1, T_k1 = find_roto_translation(P_k1_new, P_k2, t_k1+t_k)

    # # Rototranslate the P_k2 in the new frame
    # P_k2_new = R_k1 @ P_k2 + T_k1

    # # Plot the results
    # plt.figure()
    # plt.plot(P_k[0, :], P_k[1, :], 'bo', markersize=10, label='P_k')
    # plt.plot(P_k1_new[0, :], P_k1_new[1, :], 'k*', markersize=10, label='P_k1_new')
    # plt.plot(P_k2_new[0, :], P_k2_new[1, :], 'rs', markersize=10, label='P_k2_new')
    # plt.axis('equal')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    rclpy.shutdown()


if __name__ == '__main__':
    main()

