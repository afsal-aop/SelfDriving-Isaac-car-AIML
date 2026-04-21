#!/usr/bin/env python3
"""
UNIFIED DATA COLLECTOR — Drive and Save in ONE Terminal

DRIVING CONTROLS:
  W = Forward
  S = Backward
  A = Turn Left
  D = Turn Right
  Spacebar = STOP

SAVING CONTROLS:
  0 = save current frame as CLASS 0 (line centered, go straight)
  1 = save current frame as CLASS 1 (drifted right, steer left)
  2 = save current frame as CLASS 2 (drifted left, steer right)
  3 = save current frame as CLASS 3 (no line, spin in place)
  
  Q = Quit
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import sys
import termios
import tty
import threading

# Save folders for each class
CLASS_DIRS = {
    '0': os.path.expanduser("~/line_follower_ml/data/class_0_center"),
    '1': os.path.expanduser("~/line_follower_ml/data/class_1_drift_right"),
    '2': os.path.expanduser("~/line_follower_ml/data/class_2_drift_left"),
    '3': os.path.expanduser("~/line_follower_ml/data/class_3_no_line"),
}

CLASS_NAMES = {
    '0': 'CENTER    → robot goes straight',
    '1': 'DRIFT RIGHT → robot steers LEFT',
    '2': 'DRIFT LEFT  → robot steers RIGHT',
    '3': 'NO LINE   → robot spins in place',
}

# Ensure directories exist
for folder in CLASS_DIRS.values():
    os.makedirs(folder, exist_ok=True)

class UnifiedCollector(Node):
    def __init__(self):
        super().__init__('unified_collector')
        self.bridge = CvBridge()
        self.latest_image = None
        
        self.counts = {'0': 0, '1': 0, '2': 0, '3': 0}
        
        # Count existing files so we don't overwrite
        for key, folder in CLASS_DIRS.items():
            existing = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
            self.counts[key] = existing

        # Subscriber for Camera
        self.img_sub = self.create_subscription(
            Image,
            '/robot_camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher for Driving
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        print("\n========================================")
        print("  UNIFIED DATA COLLECTOR & TELEOP")
        print("========================================")
        print("DRIVE: W/A/S/D | STOP: Spacebar")
        print("SAVE:  0/1/2/3 | QUIT: Q")
        print("========================================")
        print("Current counts:")
        for k, v in self.counts.items():
            print(f"  Class {k} ({CLASS_NAMES[k]}): {v} images")
        print("========================================\n")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = cv_image.shape[:2]
        bottom = cv_image[h // 2:, :]
        self.latest_image = cv2.resize(bottom, (64, 64))

    def save_image(self, class_key):
        if self.latest_image is None:
            print("  No camera image received yet! Is Isaac Sim playing?")
            return
            
        folder = CLASS_DIRS[class_key]
        count = self.counts[class_key]
        filename = f"img_{count:06d}.jpg"
        filepath = os.path.join(folder, filename)
        
        cv2.imwrite(filepath, self.latest_image)
        self.counts[class_key] += 1
        total = sum(self.counts.values())
        print(f"  Saved class {class_key} ({CLASS_NAMES[class_key]}) | count={self.counts[class_key]} | total={total}")

    def publish_twist(self, speed, turn):
        twist = Twist()
        twist.linear.x = float(speed)
        twist.angular.z = float(turn)
        self.cmd_pub.publish(twist)


def get_key():
    """Read a single keypress from terminal without Enter"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

def main():
    rclpy.init()
    node = UnifiedCollector()

    # Spin ROS2 in background thread so camera keeps updating
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Main loop — read keypresses
    try:
        while True:
            key = get_key().lower() # Convert to lowercase to handle Caps Lock
            
            # Driving controls
            if key == 'w':
                node.publish_twist(0.2, 0.0)
            elif key == 's':
                node.publish_twist(-0.2, 0.0)
            elif key == 'a':
                node.publish_twist(0.1, 0.5)
            elif key == 'd':
                node.publish_twist(0.1, -0.5)
            elif key == ' ':
                node.publish_twist(0.0, 0.0)
                
            # Saving controls
            elif key in ('0', '1', '2', '3'):
                node.save_image(key)
                
            # Quit control
            elif key == 'q':
                print("\nQuitting. Final counts:")
                for k, v in node.counts.items():
                    print(f"  Class {k}: {v} images")
                # Stop the robot before quitting
                node.publish_twist(0.0, 0.0)
                break
                
    except KeyboardInterrupt:
        node.publish_twist(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
