#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from geometry_msgs.msg import Twist  # For publishing velocity
from nav_msgs.msg import Odometry    # For subscribing to /odom
from where_are_you_ros2.action import MoveDistance  # Import your custom action
import math

class MoveDistanceServer(Node):

    def __init__(self):
        super().__init__('move_distance_server')

        # Action server to handle the move_distance action
        self._action_server = ActionServer(
            self,
            MoveDistance,
            'move_distance',
            self.execute_callback
        )

        # Publisher for /cmd_vel (velocity commands)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for /odom to track real movement
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.current_x = 0.0
        self.current_y = 0.0
        self.get_logger().info('Move Distance Action Server is ready.')

    def odom_callback(self, msg):
        """Update current position from /odom topic."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def execute_callback(self, goal_handle):
        """Handle goal execution and publish velocity commands."""
        self.get_logger().info(f'Executing goal to x: {goal_handle.request.x_goal}, y: {goal_handle.request.y_goal}')
        
        # Feedback and result initialization
        feedback_msg = MoveDistance.Feedback()
        result_msg = MoveDistance.Result()

        goal_x = goal_handle.request.x_goal
        goal_y = goal_handle.request.y_goal

        # Calculate the distance to the goal
        total_distance = self.get_distance_to_goal(goal_x, goal_y)
        linear_speed = 0.5  # Fixed linear speed (m/s)
        tolerance = 0.1  # Stopping tolerance

        while total_distance > tolerance:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return MoveDistance.Result()

            # Calculate remaining distance and direction
            total_distance = self.get_distance_to_goal(goal_x, goal_y)
            direction_x = goal_x - self.current_x
            direction_y = goal_y - self.current_y
            distance_angle = math.atan2(direction_y, direction_x)

            # Create a Twist message to command the robot
            twist_msg = Twist()

            # Move forward if aligned with the goal direction
            twist_msg.linear.x = linear_speed

            # Publish the velocity command
            self.publisher_.publish(twist_msg)

            # Update feedback and publish
            feedback_msg.current_x = self.current_x
            feedback_msg.current_y = self.current_y
            goal_handle.publish_feedback(feedback_msg)

        # Stop the robot after reaching the goal
        self.stop_robot()

        # Set result when the goal is reached
        result_msg.x_actual = self.current_x
        result_msg.y_actual = self.current_y
        goal_handle.succeed()

        self.get_logger().info(f'Goal succeeded. Final position: x: {self.current_x}, y: {self.current_y}')
        return result_msg

    def get_distance_to_goal(self, goal_x, goal_y):
        """Calculate Euclidean distance to the goal."""
        return math.sqrt((goal_x - self.current_x)**2 + (goal_y - self.current_y)**2)

    def stop_robot(self):
        """Send a zero velocity command to stop the robot."""
        twist_msg = Twist()
        self.publisher_.publish(twist_msg)  # Publish zero velocity to stop


def main(args=None):
    rclpy.init(args=args)

    action_server = MoveDistanceServer()

    rclpy.spin(action_server)

    action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

