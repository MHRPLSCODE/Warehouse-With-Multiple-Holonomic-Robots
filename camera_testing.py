#!/usr/bin/env python3

import cv2
import time
import yaml
import signal
import subprocess
import rclpy
import numpy as np
import os

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


# ================= USER TUNING PARAMETERS =================
EXPOSURE_TIME = 200
GAIN = 5
BRIGHTNESS = 20
CONTRAST = 60
SATURATION = 60
HUE = 0
WHITE_BALANCE_TEMP = 4800
GAMMA = 120
TARGET_FPS = 25
SHOW_WINDOW = True
# ==========================================================


class CameraTester(Node):
    def __init__(self):
        super().__init__('camera_tester_node')

        self.bridge = CvBridge()
        self.CAMERA_ID = 2   # /dev/video0

        # ---------- ROS PUBLISHERS ----------
        self.image_publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        # ---------- SIGNAL HANDLING ----------
        signal.signal(signal.SIGINT, self.signal_handler)

        # ---------- APPLY HARDWARE CONTROLS ----------
        self.apply_v4l2_controls()

        # ---------- OPEN CAMERA ----------
        self.cap = cv2.VideoCapture(self.CAMERA_ID, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        # ---------- FORCE 1080p MJPG ----------
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.get_logger().info(f"Camera resolution locked to {w}x{h}")
        self.get_logger().info(f"Requested FPS={TARGET_FPS}, OpenCV reports FPS={fps}")

        self.print_camera_info()

        # ---------- LOAD CALIBRATION ----------
        calib_path = os.path.join(
            get_package_share_directory('hb_control'),
            'config',
            'camera_testing.yaml'
        )
        self.camera_info_msg = self.load_camera_info(calib_path)

        # ---------- FAST GAMMA LUT (NO HEAVY POSTPROCESS) ----------
        # Keeps quality stable and boosts FPS (your old fix_image() was slow)
        gamma = 0.8
        self.gamma_lut = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
            dtype=np.uint8
        )

        self.get_logger().info("Camera Testing Node Started")

    # =================== CAMERA CALIBRATION ===================
    def load_camera_info(self, yaml_path):
        with open(yaml_path, 'r') as f:
            calib = yaml.safe_load(f)

        msg = CameraInfo()

        msg.width = calib.get('image_width', 1920)
        msg.height = calib.get('image_height', 1080)

        msg.k = calib['camera_matrix']['data']
        msg.d = calib['distortion_coefficients']['data']
        msg.r = calib.get('rectification_matrix', {'data': [1, 0, 0, 0, 1, 0, 0, 0, 1]})['data']
        msg.p = calib['projection_matrix']['data']

        msg.distortion_model = calib.get('distortion_model', 'plumb_bob')
        msg.header.frame_id = "camera_link"

        self.get_logger().info(f"Camera calibration loaded ({msg.width}x{msg.height})")
        return msg

    # =================== HARDWARE CONTROLS ===================
    def apply_v4l2_controls(self):
        def ctl(cmd):
            subprocess.run(cmd, check=False)

        dev = f"/dev/video{self.CAMERA_ID}"

        # Force FPS at driver level (OpenCV alone often lies/ignores)
        ctl(["v4l2-ctl", "-d", dev, "--set-parm", str(TARGET_FPS)])

        ctl(["v4l2-ctl", "-d", dev, "-c", "auto_exposure=1"])
        ctl(["v4l2-ctl", "-d", dev, "-c", f"exposure_time_absolute={EXPOSURE_TIME}"])
        ctl(["v4l2-ctl", "-d", dev, "-c", f"gain={GAIN}"])
        ctl(["v4l2-ctl", "-d", dev, "-c", f"brightness={BRIGHTNESS}"])
        ctl(["v4l2-ctl", "-d", dev, "-c", f"contrast={CONTRAST}"])
        ctl(["v4l2-ctl", "-d", dev, "-c", f"saturation={SATURATION}"])
        ctl(["v4l2-ctl", "-d", dev, "-c", f"hue={HUE}"])
        ctl(["v4l2-ctl", "-d", dev, "-c", "white_balance_automatic=0"])
        ctl(["v4l2-ctl", "-d", dev, "-c", f"white_balance_temperature={WHITE_BALANCE_TEMP}"])
        ctl(["v4l2-ctl", "-d", dev, "-c", f"gamma={GAMMA}"])

    # =================== IMAGE POST-PROCESS ===================
    def fix_image(self, frame):
        # FAST: gamma LUT only (no percentiles, no float conversions)
        return cv2.LUT(frame, self.gamma_lut)

    # =================== CAMERA INFO PRINT ===================
    def print_camera_info(self):
        print("\n" + "=" * 60)
        print("Camera Properties:")
        print("=" * 60)

        props = {
            'Frame Width': cv2.CAP_PROP_FRAME_WIDTH,
            'Frame Height': cv2.CAP_PROP_FRAME_HEIGHT,
            'FPS': cv2.CAP_PROP_FPS,
            'Brightness': cv2.CAP_PROP_BRIGHTNESS,
            'Contrast': cv2.CAP_PROP_CONTRAST,
            'Saturation': cv2.CAP_PROP_SATURATION,
            'Hue': cv2.CAP_PROP_HUE,
            'Gain': cv2.CAP_PROP_GAIN,
            'Exposure': cv2.CAP_PROP_EXPOSURE,
            'Buffer Size': cv2.CAP_PROP_BUFFERSIZE,
        }

        for name, prop in props.items():
            print(f"{name:20s}: {self.cap.get(prop)}")

        print("=" * 60)

    # =================== CTRL+C ===================
    def signal_handler(self, sig, frame):
        self.cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        exit(0)

    # =================== MAIN LOOP ===================
    def run(self):
        start = time.time()
        frames = 0
        last_time = time.time()

        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = self.fix_image(frame)

            frames += 1
            now = time.time()
            fps = frames / (now - start)

            # show instantaneous fps too
            inst_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            cv2.putText(frame, f"FPS(avg): {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            cv2.putText(frame, f"FPS(inst): {inst_fps:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_publisher.publish(img_msg)

            self.camera_info_msg.header.stamp = img_msg.header.stamp
            self.camera_info_publisher.publish(self.camera_info_msg)

            if SHOW_WINDOW:
                cv2.imshow("Camera Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = CameraTester()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()