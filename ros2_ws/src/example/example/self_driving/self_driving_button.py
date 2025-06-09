#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# Button press receiver for autonomous driving

from rclpy.node import Node
from ros_robot_controller_msgs.msg import ButtonState

class ButtonPressReceiver(Node):
    def __init__(self, name):
        super().__init__(name)
        self.create_subscription(ButtonState, '/ros_robot_controller/button', self.button_callback, 10)
        self.get_logger().info('ButtonPressReceiver node started')

    def button_callback(self, msg):
        if msg.id == 1:
            self.process_button_press('Button 1', msg.state)
        elif msg.id == 2:
            self.process_button_press('Button 2', msg.state)

    def process_button_press(self, button_name, state):
        if state == 1:
            self.get_logger().info(f'{button_name} short press detected')
            # Add additional logic here for short press
        elif state == 2:
            self.get_logger().info(f'{button_name} long press detected')
            # Add additional logic here for long press

def main():
    import rclpy
    rclpy.init()
    node = ButtonPressReceiver('button_press_receiver')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()