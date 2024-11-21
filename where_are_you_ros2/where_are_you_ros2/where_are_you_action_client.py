#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav_msgs.msg import Odometry  # For reading /odom data
from sensor_msgs.msg import Range
from action_move_interfaces.action import MoveDistance  # Import custom action
from action_move_interfaces.action import Rotate
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tf_transformations
import os
import plotly.graph_objects as go



class WhereAreYou():
    def __init__(self):
        
        self.anchor_dist = 1.0
        # Initialize the matrix of distances between the anchors
        self.D_fixed = np.array([[.0, .0, .0],
                                [.0, .0, self.anchor_dist],
                                [.0, self.anchor_dist, .0]])
        

        # Initialize the estimated points P_k, P_k1, and P_k2
        self.P_k = np.zeros((2, 3))
        self.P_k1 = np.zeros((2, 3))
        self.P_k2 = np.zeros((2, 3))

        self.P_relative = np.zeros((2, 3))

        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.master, = self.ax.plot([], [], 'bo', markersize=10, label='Master')
        self.anchor1, = self.ax.plot([], [], 'k*', markersize=10, label='Anchor 1')
        self.anchor2, = self.ax.plot([], [], 'rs', markersize=10, label='Anchor 2')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_aspect('equal')
        self.ax.legend()
        self.ax.grid(True)

        self.master_x = []
        self.master_y = []
        self.anchor1_x = []
        self.anchor1_y = []
        self.anchor2_x = []
        self.anchor2_y = []


    def init_plot(self):

        self.master_x.append(self.P_k_f[0, 0])
        self.master_y.append(self.P_k_f[1, 0])
        self.anchor1_x.append(self.P_k_f[0, 1])
        self.anchor1_y.append(self.P_k_f[1, 1])
        self.anchor2_x.append(self.P_k_f[0, 2])
        self.anchor2_y.append(self.P_k_f[1, 2])
        self.master_x.append(self.P_k1_f[0, 0])
        self.master_y.append(self.P_k1_f[1, 0])
        self.master_x.append(self.P_k2_f[0, 0])
        self.master_y.append(self.P_k2_f[1, 0])

        self.master.set_data(self.master_x, self.master_y)
        self.anchor1.set_data(self.anchor1_x, self.anchor1_y)
        self.anchor2.set_data(self.anchor2_x, self.anchor2_y)
                # Start the animation
        print("master_x:", self.master_x)
        print("master_y:", self.master_y)
        print("anchor1_x:", self.anchor1_x)
        print("anchor1_y:", self.anchor1_y)
        print("anchor2_x:", self.anchor2_x)
        print("anchor2_y:", self.anchor2_y)

        plt.draw()
        print("Init plot")


    def update_plot(self, master_x_c, master_y_c):
        self.master_x.append(master_x_c)
        self.master_y.append(master_y_c)
        self.master.set_data(self.master_x, self.master_y)
        plt.draw()
        plt.show()

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

        self.init_plot()
        self.update_plot(4,5)
    
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
            '/uwb',
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
        theta = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0]) * 180 / np.pi
        print("Theta:", theta)


        

    def timer_callback(self):
        
        if self.status == 0:
            if self.flag:
                self.dist_1 = np.array([self.current_dist_1, self.current_dist_2])
                # self.D_k1 = self.compute_distance_matrix(self.current_dist_1, self.current_dist_2, self.ancor_dist)
                self.send_goal(self.t_k[0], self.t_k[1])
                self.flag = False
        if self.status == 1:
            if self.flag:
                self.send_rotation(self.theta)
                self.flag = False
        if self.status == 2:
            if self.flag:
                self.dist_2 = np.array([self.current_dist_1, self.current_dist_2])
                self.send_goal(self.t_k1[0], self.t_k1[1])
                self.flag = False
                # os.system('ros2 run ')
        if self.status == 3:
            self.get_logger().info('Both goals successfully reached! Starting relative localization')
            ## relative localization
            if self.flag:
                self.dist_3 = np.array([self.current_dist_1, self.current_dist_2])
                self.flag = False
                self.where_are_you.where_are_you(self.dist_1, self.dist_2, self.dist_3, self.t_k, self.t_k1) 
                # Find the relative position between the anchors and the robot using where are you
                
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
    # client = MoveDistanceClient()

    # # Get initial odometry before sending the goal
    # client.get_logger().info(f'Initial position: x: {client.current_x}, y: {client.current_y}')

    # rclpy.spin(client)

    # client.destroy_node()

    # test where are we
    where_are_you = WhereAreYou()

    # # Test the where are you function
    dist_1 = np.array([1,np.sqrt(2)])
    dist_2 = np.array([np.sqrt(2),1])
    dist_3 = np.array([np.sqrt(5),2])
    t_k = np.array([0.0,1.0])
    t_k1 = np.array([-1.0,0.0])

    # init the plot
    # where_are_you.init_plot()

    # Call the where are you function
    where_are_you.where_are_you(dist_1, dist_2, dist_3, t_k, t_k1) 
    
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

