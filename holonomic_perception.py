#!/usr/bin/env python3
"""
ROS2 localization node with corner-marker locking
(using EXTREME ArUco corners for homography)
"""

import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from hb_interfaces.msg import Pose2D, Poses2D
from std_msgs.msg import Float64MultiArray
# =========================================================


class PoseDetector(Node):
    def __init__(self):
        super().__init__('localization_node')

        self.bridge = CvBridge()

        # ---------- PARAMETERS ----------
        self.aruco_dict_name = 'DICT_4X4_50'

        # ---------- ALLOWED IDS ----------
        self.allowed_bot_ids = {0, 2, 4}
        self.allowed_crate_ids = {12, 21, 16, 30, 20, 14}

        self.homography_pub = self.create_publisher(Float64MultiArray,'/arena/homography',10)
        
        
        # Corner markers
        self.corner_ids = [1, 3, 5, 7]
        self.world_points_mm = {
            1: (0.0, 0.0),        # bottom-left
            3: (2438.4, 0.0),     # bottom-right
            7: (2438.4, 2438.4),  # top-right
            5: (0.0, 2438.4)      # top-left
        }

        # ---------- CORNER LOCKING ----------
        self.corner_pixel_history = {}
        self.corner_pixel_locked = {}
        self.CORNER_LOCK_FRAMES = 15
        self.CORNER_STD_THRESH = 2.0

        # ---------- ROS ----------
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10)
        self.cam_info_sub = self.create_subscription(
            CameraInfo, "/camera/camera_info", self.camera_info_callback, 10)

        self.crate_poses_pub = self.create_publisher(
            Poses2D, '/crate_pose', 10)
        self.bot_poses_pub = self.create_publisher(
            Poses2D, '/bot_pose', 10)

        # ---------- CAMERA ----------
        self.camera_matrix = None
        self.dist_coeffs = None

        # ---------- HOMOGRAPHY ----------
        self.H_matrix = None

        # ---------- ARUCO ----------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.aruco_dict_name))
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.aruco_params = cv2.aruco.DetectorParameters()

        # ---- CRITICAL FIXES ----
        self.aruco_params.minMarkerPerimeterRate = 0.025
        self.aruco_params.maxMarkerPerimeterRate = 4.0

        self.aruco_params.minCornerDistanceRate = 0.01
        self.aruco_params.minDistanceToBorder = 3

        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 40
        self.aruco_params.adaptiveThreshWinSizeStep = 4

        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.01

        self.aruco_params.errorCorrectionRate = 0.4
        self.aruco_params.polygonalApproxAccuracyRate = 0.03


        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.aruco_params)

        self.get_logger().info("✅ PoseDetector initialized")


    # =========================================================
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("📷 Camera info received")


    # =========================================================
    def pixel_to_world(self, px, py):
        if self.H_matrix is None:
            return None, None
        src = np.array([[[px, py]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H_matrix)
        return float(dst[0][0][0]), float(dst[0][0][1])


    # =========================================================
    def image_callback(self, msg):
        try:
            if self.camera_matrix is None:
                return

            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            undistorted = cv2.undistort(
                frame, self.camera_matrix, self.dist_coeffs)

            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            if ids is None:
                cv2.imshow("Detected Markers", undistorted)
                cv2.waitKey(1)
                return

            ids = ids.flatten()

            # =====================================================
            # CORNER MARKER LOCKING (EXTREME CORNERS)
            # =====================================================
            for i, marker_id in enumerate(ids):

                if marker_id not in self.corner_ids:
                    continue

                if marker_id in self.corner_pixel_locked:
                    continue

                c = corners[i][0]  # (4,2)

                # Select correct arena corner
                if marker_id == 1:      # bottom-left
                    extreme_pt = c[0]
                elif marker_id == 3:    # bottom-right
                    extreme_pt = c[1]
                elif marker_id == 7:    # top-right
                    extreme_pt = c[2]
                elif marker_id == 5:    # top-left
                    extreme_pt = c[3]
                else:
                    continue

                self.corner_pixel_history.setdefault(marker_id, []).append(extreme_pt)

                if len(self.corner_pixel_history[marker_id]) > self.CORNER_LOCK_FRAMES:
                    self.corner_pixel_history[marker_id].pop(0)

                if len(self.corner_pixel_history[marker_id]) == self.CORNER_LOCK_FRAMES:
                    pts = np.array(self.corner_pixel_history[marker_id])
                    if np.std(pts[:, 0]) < self.CORNER_STD_THRESH and \
                       np.std(pts[:, 1]) < self.CORNER_STD_THRESH:

                        self.corner_pixel_locked[marker_id] = np.mean(pts, axis=0)
                        self.get_logger().info(f"🔒 Corner {marker_id} locked")

            # =====================================================
            # HOMOGRAPHY
            # =====================================================
            if len(self.corner_pixel_locked) == 4:
                pixel_pts, world_pts = [], []

                for cid in self.corner_ids:
                    pixel_pts.append(self.corner_pixel_locked[cid])
                    world_pts.append(self.world_points_mm[cid])

                    cv2.circle(
                        undistorted,
                        tuple(self.corner_pixel_locked[cid].astype(int)),
                        8, (0, 255, 0), -1
                    )

                self.H_matrix, _ = cv2.findHomography(
                    np.array(pixel_pts, np.float32),
                    np.array(world_pts, np.float32),
                    cv2.RANSAC, 3.0)

            if self.H_matrix is None:
                cv2.imshow("Detected Markers", undistorted)
                cv2.waitKey(1)
                return
            self.H_matrix, _ = cv2.findHomography(
            np.array(pixel_pts, np.float32),
            np.array(world_pts, np.float32),
            cv2.RANSAC, 3.0)
            if self.H_matrix is not None:
                 hmsg = Float64MultiArray()
                 hmsg.data = self.H_matrix.flatten().tolist()
                 self.homography_pub.publish(hmsg)


            # =====================================================
            # DRAW & PROCESS NON-CORNER MARKERS
            # =====================================================
            crate_poses = []
            bot_poses = []

            for i, marker_id in enumerate(ids):
                if marker_id in self.corner_ids:
                    continue

                cv2.aruco.drawDetectedMarkers(
                    undistorted, [corners[i]], np.array([marker_id]))

                c = corners[i][0]
                center_px = np.mean(c, axis=0)

                x_mm, y_mm = self.pixel_to_world(center_px[0], center_px[1])
                if x_mm is None:
                    continue

                vec = c[0] - center_px
                yaw_deg = (math.degrees(math.atan2(vec[1], vec[0])) + 135.0) % 360.0

                text = f"ID:{marker_id} X:{x_mm:.0f} Y:{y_mm:.0f} Yaw:{yaw_deg:.1f}"
                cv2.putText(
                    undistorted,
                    text,
                    (int(center_px[0] + 5), int(center_px[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

                pose = Pose2D()
                pose.id = int(marker_id)
                pose.x = x_mm
                pose.y = y_mm
                pose.w = yaw_deg

                if marker_id in self.allowed_bot_ids:
                    bot_poses.append(pose)
                elif marker_id in self.allowed_crate_ids:
                    crate_poses.append(pose)
                # else: ignore marker completely


            self.publish_bot_poses(bot_poses)
            self.publish_crate_poses(crate_poses)

            cv2.imshow("Detected Markers", undistorted)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(str(e))


    # =========================================================
    def publish_crate_poses(self, poses):
        msg = Poses2D()
        msg.poses = poses
        self.crate_poses_pub.publish(msg)

    def publish_bot_poses(self, poses):
        msg = Poses2D()
        msg.poses = poses
        self.bot_poses_pub.publish(msg)


# =========================================================
def main(args=None):
    rclpy.init(args=args)
    node = PoseDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()