#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from geometry_msgs.msg import Twist  # For publishing velocity
from nav_msgs.msg import Odometry    # For subscribing to /odom
from action_move_interfaces.action import MoveDistance  # Import your custom action
from action_move_interfaces.action import Rotate
import math
import time
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import tf_transformations 

class MoveDistanceServer(Node):

    def __init__(self):
        super().__init__('move_distance_server')

        self.action_ = MutuallyExclusiveCallbackGroup()
        self.topic_ = MutuallyExclusiveCallbackGroup()

        # Action server to handle the move_distance action
        self._action_server = ActionServer(
            self,
            MoveDistance,
            'move_distance',
            self.execute_callback,
            callback_group=self.action_
        )

        self._action_server_rotate = ActionServer(
            self,
            Rotate,
            'rotate',
            self.execute_callback_rotate,
            callback_group=self.action_
        )

        # Publisher for /cmd_vel (velocity commands)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for /odom to track real movement
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10,
            callback_group=self.topic_
        )

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_angle = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.get_logger().info('Move Distance Action Server is ready.')

    def odom_callback(self, msg):
        """Update current position from /odom topic."""
        if msg.child_frame_id == 'base_footprint':
            self.current_x = msg.pose.pose.position.x
            self.current_y = msg.pose.pose.position.y
            # Convert quaternion to Euler angles
            orientation_q = msg.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (_, _, yaw) = tf_transformations.euler_from_quaternion(orientation_list)
            self.current_angle = yaw

    def execute_callback(self, goal_handle):
        """Handle goal execution and publish velocity commands."""
        self.get_logger().info(f'Executing goal distance: {goal_handle.request.target_distance}')
        
        # Feedback and result initialization
        feedback_msg = MoveDistance.Feedback()
        result_msg = MoveDistance.Result()

        goal_distance = goal_handle.request.target_distance
        print(f"Goal distance: {goal_distance}")
        self.start_x = self.current_x
        self.start_y = self.current_y

        # Calculate the distance to the goal
        total_distance = 0.0
        linear_speed = 0.2  # Fixed linear speed (m/s)
        tolerance = 0.02  # Stopping tolerance

        while total_distance < goal_distance - tolerance:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return MoveDistance.Result()

            # Calculate remaining distance and direction
            total_distance = self.get_distance_to_goal()

            # Create a Twist message to command the robot
            twist_msg = Twist()

            # Move forward if aligned with the goal direction
            twist_msg.linear.x = linear_speed

            # Publish the velocity command
            self.publisher_.publish(twist_msg)

            # # Update feedback and publish
            feedback_msg.current_x = self.current_x
            feedback_msg.current_y = self.current_y
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.05)   

        # Stop the robot after reaching the goal
        self.stop_robot()

        # Set result when the goal is reached
        result_msg.final_distance = self.get_distance_to_goal()
        goal_handle.succeed()

        print(f"Goal distance: {result_msg.final_distance}")

        self.get_logger().info(f'Goal succeeded. Final position: x: {self.current_x}, y: {self.current_y}')
        return result_msg
    
    def execute_callback_rotate(self, goal_handle):
        # Function that reads the goal and executes the action
        self.get_logger().info(f'Executing goal to angle: {goal_handle.request.rotation_angle}')

        # Feedback and result initialization
        feedback_msg = Rotate.Feedback()
        result_msg = Rotate.Result()
        goal_angle = goal_handle.request.rotation_angle * math.pi / 180  # Convert to radians
        angular_speed = 0.5  # Fixed angular speed (rad/s)
        tolerance = 0.01  # Stopping tolerance

        # Current rotated angle
        rotated_angle = 0.0

        # Starting orientation from quaternion in odom to Euler angle
        starting_angle = self.current_angle

        print(f"Goal angle: {goal_angle}")
        print(f"Rotated angle: {rotated_angle}")
        print(f"Starting angle: {starting_angle}")

        while abs(rotated_angle) < abs(goal_angle):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Rotate.Result()

            # Create a Twist message to command the robot
            twist_msg = Twist()
            # Wrap the angle to [-pi, pi]
            remaining_angle = goal_angle - rotated_angle
            remaining_angle = (remaining_angle + math.pi) % (2 * math.pi) - math.pi

            twist_msg.angular.z = angular_speed if remaining_angle > 0 else -angular_speed

            # Publish the velocity command
            self.publisher_.publish(twist_msg)

            time.sleep(0.1)

            # Calculate the rotated angle
            rotated_angle = starting_angle - self.current_angle
            rotated_angle = (rotated_angle + math.pi) % (2 * math.pi) - math.pi
            
            # Update feedback and publish
            feedback_msg.current_rotation_angle = rotated_angle
            goal_handle.publish_feedback(feedback_msg)

        # Stop the robot after reaching the goal
        self.stop_robot()

        # Set result when the goal is reached
        result_msg.actual_rotation_angle = goal_angle
        goal_handle.succeed()

        self.get_logger().info(f'Goal succeeded. Final angle: {goal_angle}')
        return result_msg

    def get_distance_to_goal(self):
        """Calculate Euclidean distance to the goal."""
        return math.sqrt((self.current_x - self.start_x)**2 + (self.current_y - self.start_y)**2)

    def stop_robot(self):
        """Send a zero velocity command to stop the robot."""
        twist_msg = Twist()
        self.publisher_.publish(twist_msg)  # Publish zero velocity to stop


def main(args=None):
    rclpy.init(args=args)

    action_server = MoveDistanceServer()

    executor = MultiThreadedExecutor()
    executor.add_node(action_server)

    try:
        executor.spin()

    except KeyboardInterrupt:
        action_server.get_logger().error("Caught keyboard interrupt")

    action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

