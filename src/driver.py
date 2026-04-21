#!/usr/bin/env python3
"""
ROBOT DRIVER — Classification ML + ArUco markers
Loads trained CNN, reads camera, decides what to do every frame.

ML decisions:
  Class 0 → go straight at normal speed
  Class 1 → steer left  (line is left, robot drifted right)
  Class 2 → steer right (line is right, robot drifted left)
  Class 3 → spin in place slowly until line is found

ArUco overrides (run at same time, take priority):
  ID 0 → STOP completely
  ID 1 → SLOW DOWN (half speed)
  ID 2 → SPEED UP  (double speed)
  ID 3 → TURN RIGHT (spin right for a moment, then resume ML)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
import os


# ── Copy of the exact same CNN class used in training ────────────────────────
class LineCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Speed and behaviour constants — tweak these if robot is too fast/slow ────
SPEED_NORMAL     = 0.25    # forward speed when going straight
SPEED_SLOW       = 0.12    # forward speed when ArUco SLOW is active
SPEED_FAST       = 0.45    # forward speed when ArUco SPEEDUP is active
STEER_SOFT       = 0.4     # angular.z when correcting drift slightly
STEER_HARD       = 0.8     # angular.z when line is far off center
SPIN_SPEED       = 0.5     # angular.z when spinning to search for line (class 3)
ARUCO_HOLD_FRAMES = 45     # how many frames to hold an ArUco command after seeing it
TURNRIGHT_FRAMES  = 35     # how many frames to spin right for ID 3 marker

# ArUco ID → command name
ARUCO_COMMANDS = {0: "STOP", 1: "SLOW", 2: "SPEEDUP", 3: "TURNRIGHT"}


class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')

        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the trained model
        model_path = os.path.expanduser("~/line_follower_ml/model.pth")
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model not found at {model_path}!")
            self.get_logger().error("Run train_model.py first!")
            raise FileNotFoundError(model_path)

        self.model = LineCNN(num_classes=4).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.get_logger().info(f"Model loaded from {model_path}")

        # Image preprocessing — MUST match training exactly
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std= [0.5, 0.5, 0.5])
        ])

        # ArUco detector setup
        aruco_dict         = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_detector = aruco.ArucoDetector(
            aruco_dict, aruco.DetectorParameters())

        # State
        self.aruco_command      = None   # active ArUco command string or None
        self.aruco_frames_left  = 0      # countdown: frames remaining for this command
        self.turnright_frames   = 0      # countdown: frames left in TURNRIGHT spin
        self.current_speed_mult = 1.0    # speed multiplier from ArUco (1=normal)

        # ROS2
        self.img_sub = self.create_subscription(
            Image, '/robot_camera/image_raw', self.on_image, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("Driver started. Robot will follow the line.")
        self.get_logger().info(
            "ArUco: ID0=STOP | ID1=SLOW | ID2=SPEEDUP | ID3=TURNRIGHT")

    # ── Called every time a new camera frame arrives ──────────────────────────
    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w  = frame.shape[:2]

        # ── Step 1: Check for ArUco markers (uses FULL image, better detection) ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is not None:
            marker_id = int(ids[0][0])
            if marker_id in ARUCO_COMMANDS:
                cmd = ARUCO_COMMANDS[marker_id]
                self.get_logger().info(f"ArUco ID {marker_id} → {cmd}")
                self.aruco_command     = cmd
                self.aruco_frames_left = ARUCO_HOLD_FRAMES

                if cmd == "SLOW":
                    self.current_speed_mult = 0.5
                elif cmd == "SPEEDUP":
                    self.current_speed_mult = 1.8
                elif cmd == "TURNRIGHT":
                    self.turnright_frames   = TURNRIGHT_FRAMES
                    self.current_speed_mult = 1.0
                elif cmd == "STOP":
                    self.current_speed_mult = 1.0

        # ── Step 2: Count down ArUco timer ──────────────────────────────────
        if self.aruco_frames_left > 0:
            self.aruco_frames_left -= 1
        else:
            # Command expired — reset to normal
            self.aruco_command      = None
            self.current_speed_mult = 1.0

        # ── Step 3: Handle STOP — highest priority, exit early ───────────────
        if self.aruco_command == "STOP":
            self.send_cmd(0.0, 0.0)
            return

        # ── Step 4: Handle TURNRIGHT spin — override ML ──────────────────────
        if self.turnright_frames > 0:
            self.turnright_frames -= 1
            # Spin right: negative angular.z = clockwise in ROS
            self.send_cmd(0.0, -SPIN_SPEED)
            return

        # ── Step 5: Run ML model on bottom half of image ─────────────────────
        bottom     = frame[h // 2:, :]           # crop bottom half
        bottom     = cv2.resize(bottom, (64, 64)) # resize to model input size
        pil_img    = PILImage.fromarray(
            cv2.cvtColor(bottom, cv2.COLOR_BGR2RGB))
        tensor     = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits     = self.model(tensor)           # shape: [1, 4]
            pred_class = logits.argmax(dim=1).item()  # 0, 1, 2, or 3

        self.get_logger().debug(f"ML class: {pred_class}")

        # ── Step 6: Convert class → speed and steering ───────────────────────
        speed   = SPEED_NORMAL * self.current_speed_mult

        if pred_class == 0:
            # Line is centered → go straight
            linear  = speed
            angular = 0.0

        elif pred_class == 1:
            # Drifted right, line is on left → steer left (positive angular.z)
            linear  = speed * 0.7      # slow slightly while correcting
            angular = STEER_SOFT

        elif pred_class == 2:
            # Drifted left, line is on right → steer right (negative angular.z)
            linear  = speed * 0.7
            angular = -STEER_SOFT

        elif pred_class == 3:
            # No line visible → stop forward, spin right slowly to search
            linear  = 0.0
            angular = -SPIN_SPEED      # spin right to find the line

        else:
            linear  = 0.0
            angular = 0.0

        self.send_cmd(linear, angular)

    # ── Helper ────────────────────────────────────────────────────────────────
    def send_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    try:
        node = RobotDriver()
    except FileNotFoundError:
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping robot...")
        node.send_cmd(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
