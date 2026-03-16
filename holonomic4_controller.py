
#!/usr/bin/env python3

#from numpy import int32
#from build.eyantrasim_msgs.rosidl_generator_py.eyantrasim_msgs import msg
import rclpy
from rclpy.node import Node
from hb_interfaces.srv import Attach
from hb_interfaces.msg import Poses2D, BotCmd, BotCmdArray
import math

from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Int32


from hb_control.path_planner import AStarPlanner

from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseArray

from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray



def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))

def clamp(a, lo, hi):
        return max(lo, min(hi, a))


class PID:
    def __init__(self, kp, ki, kd, limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.i = 0.0
        self.prev = 0.0
        self.prev_time = None

    def reset(self):
        self.i = 0.0
        self.prev = 0.0
        self.prev_time = None

    def step(self, error, t):
        dt = 1e-3 if self.prev_time is None else max(t - self.prev_time, 1e-3)
        self.i += error * dt
        d = (error - self.prev) / dt
        self.prev = error
        self.prev_time = t
        u = self.kp * error + self.ki * self.i + self.kd * d
        return max(-self.limit, min(self.limit, u))


class HolonomicController(Node):
    def __init__(self):
        super().__init__('holonomic4_controller')

        self.crates_delivered = 0
        self.MAX_CRATES = 2

        # =================== SPECIFIC TO BOT =================================
        # ---------------- IDs ----------------     # UNIQUE TO BOT
        self.BOT_ID = 4
        self.IGNORE_BOT_ID = 0
        self.assigned_crate_id = None
        
        self.stack_flag = False
        print("STACK FLAG: ", self.stack_flag)
        
        #self.BOX_ID_1 = 30
        #self.BOX_ID_2 = None
        #self.HALF_DONE = False
        # ---------------- Drop Zone ----------------   # UNIQUE TO BOT
        #self.DROP_X = 723.0
        #self.DROP_Y = 1797.2
        
        # ---------------- Start Zone ----------------   # UNIQUE TO BOT
        self.start_x = 1239.0   # UNIQUE TO BOT
        self.start_y = 115.0    # UNIQUE TO BOT

        # ---------------- compute_pickup_point OFFSET ----------------   # UNIQUE TO BOT
        self.OFFSET = 0.0                      # UNIQUE TO BOT

        # ---------------- ALIGN_AT_DROP angle ----------------   # UNIQUE TO BOT
        #self.drop_angle = 0.0

        # ---------------- PID ----------------     # UNIQUE TO BOT
        self.pid_limiter = 40.0

        self.pid_x = PID(6.0, 0.0, 0.9, self.pid_limiter)
        self.pid_y = PID(6.0, 0.0, 0.9, self.pid_limiter)



        # ---------------- arm angles ----------------     # UNIQUE TO BOT
        self.default_elbow = 90.0
        self.pickup_elbow = 90.0
        self.default_base = 65.0
        self.drop_base =  25.0
        self.rotate_base = 65.0 
        self.stack_lift_base = 90.0
        self.stack_lift_elbow = 40.0
        self.stack_drop_base = 70.0
        self.stack_drop_elbow = 40.0
        self.pbase = None
        self.pelbow = None

        # Start in a realistic pose
        self.last_base = 20.0
        self.last_elbow = 20.0

        # ---------------- State ----------------
        self.state = "PATH_TO_CRATE"
        self.path = []
        self.wp_idx = 0
    
        # ---------------- Poses ----------------
        self.bot_pose = None
        self.ibot_pose = None
        self.crate_pose = None

        self.tx = None
        self.ty = None

        self.last_rot = None

        self.lift_start_time = None
        self.lift_tune = 5.0


        # ---------------- Geometry ----------------
        self.MARKER_SIZE = 40.0
        self.WP_RADIUS   = 40.0    # waypoint acceptance (grid cells)

        self.STOP_RADIUS = 90.0
        self.BRAKE_RADIUS = 10.0   # start slowing down here

        self.pickup_start_time = None
        self.PICKUP_DELAY = 1.0  # seconds (tune this)


        # -------- QoS (latched path) --------
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )


        # ---------------- ROS ----------------
        self.create_subscription(Poses2D, '/bot_pose', self.bot_cb, 10)
        self.create_subscription(Poses2D, '/crate_pose', self.crate_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            "/planner/path_to_crate_bot4",
            self.crate_path_cb,
            latched_qos
        )

        self.create_subscription(
            Int32,
            "/planner/crate_id_bot4",
            self.crate_id_cb,
            latched_qos
        )

        self.state_pub = self.create_publisher( 
           Int32 ,
          "/planner/path_done",
          10
        )

        self.create_subscription(
            Float64MultiArray,
            "/planner/path_to_drop_bot4",
            self.drop_path_cb,
            latched_qos
        )

        self.create_subscription(
            Float64MultiArray,
            "/planner/path_to_home_bot4",
            self.return_path_cb,
            latched_qos
        )
        #----------------IR Sensor Subscription----------------
        self.state_ir = True
        self.prev_ir = True
        self.create_subscription(Bool, '/bot4/ir', self.ir_cb, 10)      #UNIQUE TO BOT
        self.cmd_pub = self.create_publisher(
            BotCmdArray,
            '/bot_cmd',
            10
        )

        self.grid_pub = self.create_publisher(OccupancyGrid, '/planner/grid', 1)
        self.path_pub = self.create_publisher(Path, '/planner/path', 1)
        self.sg_pub = self.create_publisher(PoseArray, '/planner/start_goal', 1)


        self.planner = AStarPlanner()
        self.path_crate = []
        self.path_drop = []
        self.path_return = []
        self.wp_idx_crate = 0
        self.wp_idx_drop = 0
        self.wp_idx_return = 0
        self.last_crates = []



        # ---------------- Attach Service Client ----------------
        self.attach_cli = self.create_client(Attach, '/attach')
        while not self.attach_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /attach service...")

        # ---------------- Timer ----------------
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("✅ Holonomic Controller Ready")

    # =========================================================
    def bot_cb(self, msg):
        for p in msg.poses:
            if p.id == self.BOT_ID:
                self.bot_pose = p
            if p.id == self.IGNORE_BOT_ID:
                self.ibot_pose = p


    # def crate_cb(self, msg):
    #     if self.state not in ["PATH_TO_CRATE", "MOVE_TO_CRATE", "ALIGN_TO_CRATE"]:
    #         return
    #     if not msg.poses:
    #         return
    #     self.crate_pose = msg.poses[0]   # planner already assigned

    def crate_cb(self, msg):
        if self.assigned_crate_id is None:
            return

        for p in msg.poses:
            if p.id == self.assigned_crate_id:
                self.crate_pose = p
                return



    def crate_id_cb(self, msg):
        self.assigned_crate_id = msg.data
        self.get_logger().info(f"Assigned crate ID = {self.assigned_crate_id}")


    def crate_path_cb(self, msg):
        d = msg.data
        print(d)
        if len(d) >= 2:
            self.path_crate = [(d[i], d[i+1]) for i in range(0, len(d), 2)]
            self.wp_idx_crate = 0
            self.pid_x.reset()
            self.pid_y.reset()
            self.state = "MOVE_TO_CRATE"
            self.get_logger().info(f"📍 Crate Path received: {len(self.path_crate)} waypoints")
        
    def drop_path_cb(self, msg):
        d = msg.data
        if len(d) >= 2:
            self.path_drop = [(d[i], d[i+1]) for i in range(0, len(d), 2)]
            self.wp_idx_drop = 0
            self.pid_x.reset()
            self.pid_y.reset()
            # self.state = "MOVE_TO_CRATE"
            self.get_logger().info(f"📍 Drop Path received: {len(self.path_drop)} waypoints")

    def return_path_cb(self, msg):
        d = msg.data
        if len(d) >= 2:
            self.path_return = [(d[i], d[i+1]) for i in range(0, len(d), 2)]
            self.wp_idx_return = 0
            self.pid_x.reset()
            self.pid_y.reset()
            # self.state = "MOVE_TO_CRATE"
            self.get_logger().info(f"📍 Return Path received: {len(self.path_return)} waypoints")


    def notify_planner(self):
        msg = Int32()
        msg.data = self.BOT_ID
        self.state_pub.publish(msg)

    
    # =========================================================
    # IK (stable + permissive)
    # =========================================================

    def arm_ik_safe(
        self,
        x_cm, y_cm,
        last_base, last_elbow,
        L1=7.0, L2=7.5
    ):
        r = math.hypot(x_cm, y_cm)

        r_min = abs(L1 - L2) + 0.5
        r_max = (L1 + L2) - 0.5

        clamped_r = False

        if r < r_min:
            scale = r_min / max(r, 1e-6)
            x_cm *= scale
            y_cm *= scale
            r = r_min
            clamped_r = True

        elif r > r_max:
            scale = r_max / r
            x_cm *= scale
            y_cm *= scale
            r = r_max
            clamped_r = True

        cos_t2 = (r*r - L1*L1 - L2*L2) / (2 * L1 * L2)
        cos_t2 = clamp(cos_t2, -1.0, 1.0)

        # elbow-down
        t2 = -math.acos(cos_t2)

        t1 = math.atan2(y_cm, x_cm) - math.atan2(
            L2 * math.sin(t2),
            L1 + L2 * math.cos(t2)
        )

        base = math.degrees(t1)
        elbow = 90.0 + math.degrees(t2)

        # 🔑 CRITICAL FIX: clamp angles, do NOT reject
        base = clamp(base, 0.0, 180.0)
        elbow = clamp(elbow, 0.0, 180.0)

        return base, elbow, clamped_r

    #----------------IR Sensor Callback----------------
    def ir_cb(self, msg):
        self.state_ir = msg.data
        # self.get_logger().info(f"IR = {self.state_ir}")
        self.state_ir=msg.data
        # self.prev_ir=self.state_ir
    def compute_pickup_point(self, crate):
        yaw = math.radians(crate.w)

        MARKER_SIZE = self.MARKER_SIZE      
        OFFSET = self.OFFSET              

        # ---- shift from corner to center ----
        dx = MARKER_SIZE / 2.0
        dy = MARKER_SIZE / 2.0

        cx = crate.x + dx * math.cos(yaw) - dy * math.sin(yaw)
        cy = crate.y + dx * math.sin(yaw) + dy * math.cos(yaw)

        # ---- apply offset along marker normal ----
        px = cx + OFFSET*math.sin(yaw)
        py = cy - OFFSET*math.cos(yaw)
        return px, py

    
    def publish_planner_debug(self, grid, path, start, goal):
        # -------- GRID --------
        og = OccupancyGrid()
        og.info.resolution = self.planner.resolution
        og.info.width = self.planner.width
        og.info.height = self.planner.height
        og.data = [cell for row in grid for cell in row]
        self.grid_pub.publish(og)

        # -------- PATH --------
        p = Path()
        for x, y in path:
            ps = PoseStamped()
            ps.pose.position.x = x
            ps.pose.position.y = y
            p.poses.append(ps)
        self.path_pub.publish(p)

        # -------- START + GOAL --------
        pa = PoseArray()
        for x, y in [start, goal]:
            pose = PoseStamped().pose
            pose.position.x = x
            pose.position.y = y
            pa.poses.append(pose)
        self.sg_pub.publish(pa)



    # =========================================================
    def publish_cmd(self, m1, m2, m3, a1, a2):
        cmd = BotCmd()
        cmd.id = self.BOT_ID
        cmd.m1 = float(m1)
        cmd.m2 = float(m2)
        cmd.m3 = float(m3)
        cmd.base = float(a1)
        cmd.elbow = float(a2)

        msg = BotCmdArray()
        msg.cmds.append(cmd)
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        if(self.stack_flag):
            self.publish_cmd(0.0, 0.0, 0.0, self.stack_lift_base, self.stack_lift_elbow)
        else:
            self.publish_cmd(0.0, 0.0, 0.0, self.default_base, self.default_elbow)

    # =========================================================
    def drive_to(self, tx, ty, final):
        bx, by = self.bot_pose.x, self.bot_pose.y
        dx, dy = tx - bx, ty - by

        dist = math.hypot(dx, dy)

        # bdx, bdy = self.ibot_pose.x - bx, self.ibot_pose.y - by
        # bdist = math.hypot(bdx, bdy)
        # if (bdist < 300):
        #     self.stop_robot()
        #     return False

        if final and dist < self.STOP_RADIUS:
            if(self.stack_flag):
                self.publish_cmd(0.0, 0.0, 0.0, self.stack_lift_base, self.stack_lift_elbow)
            else:
                self.publish_cmd(0.0, 0.0, 0.0, self.default_base, self.default_elbow)
            return True

        if not final and dist < self.WP_RADIUS:
            return True

        # ---------- STOP ZONE ----------
        # if edge_dist <= self.STOP_RADIUS:
        #     # self.stop_robot()
        #     return True

        # ---------- BRAKE SCALE ----------
        if dist < self.BRAKE_RADIUS:
            scale = dist / self.BRAKE_RADIUS
            scale = max(0.2, min(1.0, scale))  # prevent stall
        else:
            scale = 1.0

        scale = 1.0

        # ---------- PID ----------
        now = self.get_clock().now().nanoseconds / 1e9
        vx = self.pid_x.step(dx, now) * scale
        vy = self.pid_y.step(dy, now) * scale

        yaw = math.radians(self.bot_pose.w)
        c, s = math.cos(yaw), math.sin(yaw)

        vx_r = c * vx + s * vy
        vy_r = -s * vx + c * vy

        m1 = -math.sin(math.radians(90))  * vx_r + math.cos(math.radians(90))  * vy_r
        m2 = -math.sin(math.radians(210)) * vx_r + math.cos(math.radians(210)) * vy_r
        m3 = -math.sin(math.radians(330)) * vx_r + math.cos(math.radians(330)) * vy_r

        if(self.stack_flag):
            self.publish_cmd(m1, m2, m3, self.stack_lift_base, self.stack_lift_elbow)
        else:
            self.publish_cmd(m1, m2, m3, 65.0, 90.0)
        # self.publish_cmd(m1, m2, m3, 30.0, 0.0)
        return False


    # =========================================================
    def call_attach(self, value: bool):
        req = Attach.Request()
        req.bot_id = self.BOT_ID
        req.attach = value
        self.attach_cli.call_async(req)
        self.get_logger().info(f"{'ATTACH' if value else 'DETACH'} command sent")

    # =========================================================

    # =========================================================
    def control_loop(self):
        if self.bot_pose is None:
            return

        # if self.state == "START":
        #     if self.crate_pose:
        #         self.pid_x.reset()
        #         self.pid_y.reset()
        #         self.state = "PATH_TO_CRATE"
        #         self.get_logger().info("START  STATE DONE")
        #         self.get_logger().info("PATH_TO_CRATE STATE START")

        if self.state == "PATH_TO_CRATE":
            if self.path_crate:
                self.state = "MOVE_TO_CRATE"
                print("PATH_TO_CRATE → MOVE_TO_CRATE")
            return

            # # ---------- compute left-facing pickup point ----------
            # if(self.crate_pose == None):
            #     return
            # pickup = self.compute_pickup_point(self.crate_pose)

            # # ---------- build obstacle grid (ignore target crate) ----------
            # grid = self.planner.build_grid(
            #     self.last_crates,
            #     ignore_id= (self.BOX_ID_2 if self.HALF_DONE else self.BOX_ID_1),
            #     bot_id = self.BOT_ID
            # )

            # # ---------- plan A* path ----------
            # self.path = self.planner.plan(
            #     start=(self.bot_pose.x, self.bot_pose.y),
            #     goal=pickup,
            #     grid=grid
            # )

            # # ---------- publish debug info ----------
            # self.publish_planner_debug(
            #     grid=grid,
            #     path=self.path,
            #     start=(self.bot_pose.x, self.bot_pose.y),
            #     goal=pickup
            # )

            # # ---------- safety check ----------
            # if not self.path:
            #     self.get_logger().error("❌ A* failed to find path to crate")
            #     self.stop_robot()
            #     return

            # self.wp_idx = 0


            # self.get_logger().info(
            #     f"Planner debug: "
            #     f"start=({self.bot_pose.x:.1f},{self.bot_pose.y:.1f}) "
            #     f"goal=({pickup[0]:.1f},{pickup[1]:.1f}) "
            #     f"path_len={len(self.path)}"
            # )

            # self.state = "MOVE_TO_CRATE"

            # self.get_logger().info("PATH_TO_CRATE → MOVE_TO_CRATE")


        elif self.state == "MOVE_TO_CRATE":
            # wx, wy = self.path[self.wp_idx]
            # bx, by = self.bot_pose.x, self.bot_pose.y
            # dx, dy = self.crate_pose.x - bx, self.crate_pose.y - by
            # center_dist = math.hypot(dx, dy)
            # if self.drive_to(wx, wy, self.drop_base, self.default_elbow):
            #     self.BRAKE_RADIUS=1.0
            #     self.wp_idx += 1
            #     if self.wp_idx >= len(self.path) or ((self.wp_idx >= (len(self.path) - 2)) and self.state_ir == False) or (center_dist < (self.STOP_RADIUS + 61.0)):
            #         if center_dist< (self.STOP_RADIUS + 55.0):
            #             print("Centre Distance to Crate Marker:",center_dist)
            #             # self.publish_cmd(10.0, 10.0, 10.0, self.drop_base, self.default_elbow)
            #         if self.wp_idx >= len(self.path):
            #             print("Final Waypoint reached")
            #         if (self.state_ir==False):
            #             print("IR Sensor Triggered")    
            #         print(self.wp_idx)
            #         self.tx, self.ty = self.crate_pose.x, self.crate_pose.y
            #         self.get_logger().info(f"WP {self.wp_idx}: dist={math.hypot(wx-self.bot_pose.x, wy-self.bot_pose.y):.1f}")
                    # self.stop_robot()
                    # self.state = "ALIGN_TO_CRATE"
                    # self.get_logger().info("MOVE_TO_CRATE STATE DONE")
                    # self.get_logger().info("ALIGN_TO_CRATE STATE START")

            if len(self.path_crate) > 0:
                if self.wp_idx_crate == len(self.path_crate) - 10 :
                    print("Correcting orientation")
                    err = self.bot_pose.w - 0.0
                    if (err > 0.2):
                        print("Orientation corrected")
                        self.wp_idx_crate += 1
                        return
                    K = 20.0
                    rot = K * err

                    MIN_ROT = 10.0
                    if abs(rot) < MIN_ROT:
                        rot = math.copysign(MIN_ROT, rot)

                    # omni-assisted spin (DO NOT change this)
                    self.publish_cmd(
                        rot, rot, rot,
                        self.default_base,
                        self.default_elbow
                    )
                elif self.wp_idx_crate >= len(self.path_crate):
                    if self.crate_pose is None:
                        return 
                    self.stop_robot()
                    self.pid_limiter = 20.0
                    print("PID LIM: ", self.pid_limiter)
                    self.state = "ALIGN_TO_CRATE"
                    self.tx, self.ty = self.crate_pose.x, self.crate_pose.y
                    self.get_logger().info("MOVE_TO_CRATE STATE DONE")
                    self.get_logger().info("ALIGN_TO_CRATE STATE START")
                    return

            tx, ty = self.path_crate[self.wp_idx_crate]
            final = (self.wp_idx_crate == len(self.path_crate) - 1)

            if self.wp_idx_crate == len(self.path_crate) - 2 :
                self.pid_limiter = 10.0
                print("PID LIM: ", self.pid_limiter)

            if self.drive_to(tx, ty, final):
                self.wp_idx_crate += 1
                

        elif self.state == "ALIGN_TO_CRATE":

            print("Inside ALIGN_TO_CRATE")

            # ---------------------------------
            # IR FALLING EDGE = alignment found
            # ---------------------------------
            if self.prev_ir == True and self.state_ir == False:

                print("IR EDGE DETECTED")

                self.stop_robot()

                # move arm immediately to rotate posture
                self.publish_cmd(
                    0.0, 0.0, 0.0,
                    self.rotate_base,
                    self.default_elbow
                )

                self.pid_x.reset()
                self.pid_y.reset()
                self.pickup_start_time=None 
                self.state = "PICKUP"
                self.get_logger().info("ALIGN_TO_CRATE DONE")
                self.get_logger().info("PICKUP STATE START")

                self.prev_ir = self.state_ir
                return

            # ---------------------------------
            # CONTINUOUS SEARCH SPIN
            # (NO RETURNS ABOVE THIS)
            # ---------------------------------
            SEARCH_ROT = 18.0

            self.publish_cmd(
                SEARCH_ROT,
                SEARCH_ROT,
                SEARCH_ROT,
                self.rotate_base,
                self.default_elbow
            )

            self.prev_ir = self.state_ir

        elif self.state == "ARM_ADJUSTMENT":

            bx, by = self.bot_pose.x, self.bot_pose.y
            cx, cy = self.crate_pose.x, self.crate_pose.y

            dist_mm = math.hypot(cx - bx, cy - by)

            x_cm = (dist_mm / 10.0) - 8.2
            y_cm = 5.0 - 15.0

            base, elbow, clamped_r = self.arm_ik_safe(
                x_cm, y_cm,
                self.last_base, self.last_elbow
            )

            self.last_base = base
            self.last_elbow = elbow

            self.get_logger().info(
                f"x={x_cm:.2f} y={y_cm:.2f} | "
                f"base={base:.1f} elbow={elbow:.1f} | "
                f"{'R_CLAMP' if clamped_r else 'OK'}",
                throttle_duration_sec=1.0
            )

            # cmd = BotCmd()
            # cmd.id = self.BOT_ID
            # cmd.m1 = 0.0
            # cmd.m2 = 0.0
            # cmd.m3 = 0.0
            # cmd.base = 10.0
            # cmd.elbow = 7.0
            # self.pbase = base
            # self.pelbow = elbow

            # msg = BotCmdArray()
            # msg.cmds.append(cmd)
            # self.cmd_pub.publish(msg)

            # base_angle= 5.0
            # elbow_angle= 7.0
            
            # # Publish the calculated angles
            # self.publish_cmd(
            #     0.0, 0.0, 0.0,  # no movement
            #     base_angle,
            #     elbow_angle
            # )
            
            # Move to PICKUP state
            self.state = "PICKUP"
            self.get_logger().info("ARM_ADJUSTMENT STATE DONE")
            self.get_logger().info("PICKUP STATE START")

        elif self.state == "PICKUP":
            if self.pickup_start_time is None:
                # first entry into PICKUP - use calculated angles
                self.pickup_start_time = self.get_clock().now().nanoseconds / 1e9
                self.publish_cmd(
                    0, 0, 0,
                    0.0,
                    90.0
                )
                
                return

            now = self.get_clock().now().nanoseconds / 1e9

            if now - self.pickup_start_time < self.PICKUP_DELAY:
                # keep holding position with calculated angles
                self.publish_cmd(
                    0, 0, 0,
                    0.0,
                    90.0
                )
                return
            
            
            self.call_attach(True)

            # delay done → move on
            self.notify_planner()
            print("Notified the planner")
            self.pickup_start_time = None
            self.state = "PATH_TO_DROP"
            self.get_logger().info("PICKUP STATE DONE")
            self.get_logger().info("PATH_TO_DROP STATE START")

        # elif self.state == "LIFT_CRATE":
        #     now = self.get_clock().now().nanoseconds / 1e9
        #     if self.lift_start_time is None:
        #         self.lift_start_time = self.get_clock().now().nanoseconds / 1e9
        #     # desired_lift_base = 60.0

        #     # elapsed_lift_time = now - self.lift_start_time
        #     # sent_base_angle = 0.0 + elapsed_lift_time * self.lift_tune
        #     # if abs(sent_base_angle) <= desired_lift_base :
        #     #     if sent_base_angle < 90:
        #     #         self.publish_cmd(
        #     #             0, 0, 0,
        #     #             sent_base_angle,
        #     #             10.0
        #     #         )
        #     #         print("Sent lift angle = ", sent_base_angle)
                
        #     #     return
        #     # print("Final sent lift angle = ", sent_base_angle)

        #     if now - self.lift_start_time < 0.5 :
        #         self.publish_cmd(
        #                 0, 0, 0,
        #                 60.0,
        #                 0.0
        #             )
        #         print("Publishing base lift angle")
        #         return

        #     self.pid_x.reset()
        #     self.pid_y.reset()
        #     self.STOP_RADIUS = 10
        #     self.lift_start_time = None
        #     self.stack_flag = True
        #     print("STACK FLAG: ", self.stack_flag)
        #     self.state = "PATH_TO_DROP"
        #     self.get_logger().info("LIFT_CRATE STATE DONE")
        #     self.get_logger().info("PATH_TO_DROP STATE START")



        elif self.state == "PATH_TO_DROP":
             # WAIT until planner actually sends a path
            if len(self.path_drop) == 0:
                return
            self.wp_idx_drop = 0
            self.state = "MOVE_TO_DROP"
            # self.stack_flag = True
            self.get_logger().info("PATH_TO_DROP → MOVE_TO_DROP")
            return


            # # ---------- build obstacle grid (crate is attached, ignore all) ----------
            # grid = self.planner.build_grid(
            #     self.last_crates,
            #     ignore_id=None,
            #     bot_id = self.BOT_ID
            # )

            # # ---------- plan A* path to drop zone ----------
            # self.path = self.planner.plan(
            #     start=(self.bot_pose.x, self.bot_pose.y),
            #     goal=(self.DROP_X, self.DROP_Y),
            #     grid=grid
            # )

            # # ---------- publish debug info ----------
            # self.publish_planner_debug(
            #     grid=grid,
            #     path=self.path,
            #     start=(self.bot_pose.x, self.bot_pose.y),
            #     goal=(self.DROP_X, self.DROP_Y)
            # )

            # # ---------- safety check ----------
            # if not self.path:
            #     self.get_logger().error("❌ A* failed to find path to DROP")
            #     self.stop_robot()
            #     return

            # self.wp_idx = 0
            # self.state = "MOVE_TO_DROP"

            # self.get_logger().info("PATH_TO_DROP → MOVE_TO_DROP")


        elif self.state == "MOVE_TO_DROP":
            # wx, wy = self.path[self.wp_idx]
            # if self.drive_to(wx, wy, self.drop_base, self.default_elbow):
            #     print("waypoint reached")
            #     self.wp_idx += 1
            #     if self.wp_idx >= len(self.path):
                    
            #         self.STOP_RADIUS = 0.0
            #         self.state = "ALIGN_AT_DROP"
            #         self.get_logger().info("MOVE_TO_DROP STATE DONE")
            #         self.get_logger().info("ALIGN_AT_DROP STATE START")

            if len(self.path_drop)>0:
                if self.wp_idx_drop == len(self.path_drop) - 1 :
                    print("Correcting orientation")
                    err = self.bot_pose.w - 0.0
                    if (err > 0.2):
                        self.wp_idx_drop += 1
                        print("Orientation corrected")
                        return
                    K = 20.0
                    rot = K * err

                    MIN_ROT = 10.0
                    if abs(rot) < MIN_ROT:
                        rot = math.copysign(MIN_ROT, rot)

                    # omni-assisted spin (DO NOT change this)
                    self.publish_cmd(
                        rot, rot, rot,
                        self.default_base,
                        self.default_elbow
                    )
                elif self.wp_idx_drop >= len(self.path_drop):
                    print("lenght of drop path:",len(self.path_drop))
                    self.STOP_RADIUS=25.0
                    # self.stop_robot()
                    if(self.stack_flag):
                        self.publish_cmd(0.0, 0.0, 0.0, self.stack_lift_base, self.stack_lift_elbow)
                    else:
                        self.publish_cmd(0.0, 0.0, 0.0, self.default_base, self.default_elbow)
                    self.state = "ALIGN_AT_DROP"
                    self.pid_limiter = 20.0
                    print("PID LIM: ", self.pid_limiter)
                    #self.tx, self.ty = self.crate_pose.x, self.crate_pose.y
                    self.get_logger().info("MOVE_TO_DROP STATE DONE")
                    self.get_logger().info("ALIGN_AT_DROP STATE START")
                    return

            tx, ty = self.path_drop[self.wp_idx_drop]
            final = (self.wp_idx_drop == len(self.path_drop) - 1)

            if self.wp_idx_drop == len(self.path_drop) - 2 :
                self.pid_limiter = 10.0
                print("PID LIM: ", self.pid_limiter)

            if self.drive_to(tx, ty, final):
                self.wp_idx_drop += 1
                


        elif self.state == "ALIGN_AT_DROP":
            byaw = math.radians(self.bot_pose.w)

            # -------- derive desired yaw from FINAL DROP PATH ----------
            # if len(self.path_drop) >= 2:
            #     x1, y1 = self.path_drop[-2]
            #     x2, y2 = self.path_drop[-1]

            #     # direction of final approach segment
            #     path_yaw = math.atan2(y2 - y1, x2 - x1)

            #     # robot must face INTO the drop cell (normal to path)
            #     desired_yaw = wrap_angle(path_yaw - math.pi / 2)
            # else:
            #     # fallback: hold current yaw
            #     if len(self.path_drop)<2:
            #         self.get_logger().info("Insufficient drop path length for alignment")
            #         return
            #     desired_yaw = byaw

            # err = wrap_angle(desired_yaw - byaw)
            desired_yaw = math.radians(270.0)
            err = wrap_angle(desired_yaw - byaw)

            # -------- aligned ----------
            if abs(err) < math.radians(12.0):
                if(self.stack_flag):
                    self.publish_cmd(0.0, 0.0, 0.0, self.stack_lift_base, self.stack_lift_elbow)
                else:
                    self.publish_cmd(
                        0.0, 0.0, 0.0,
                        self.default_base,
                        self.default_elbow
                    )
                self.pid_x.reset()
                self.pid_y.reset()
                self.stack_flag = False
                print("STACK FLAG: ", self.stack_flag)
                self.state = "DROP"
                self.get_logger().info("ALIGN_AT_DROP DONE")
                self.get_logger().info("DROP START")
                return

            # -------- rotate (fast + stable) ----------
            K = 20.0
            rot = K * err

            MIN_ROT = 8.0
            if abs(rot) < MIN_ROT:
                rot = math.copysign(MIN_ROT, rot)

            # omni-assisted spin (DO NOT change this)
            if(self.stack_flag):
                self.publish_cmd(
                    rot, rot, rot,
                    self.stack_lift_base,
                    self.stack_lift_elbow
                )
            else:
                self.publish_cmd(
                    rot, rot, rot,
                    self.default_base,
                    self.default_elbow
                )
            

        elif self.state == "DROP":

            if self.pickup_start_time is None:
                # first entry into PICKUP
                self.pickup_start_time = self.get_clock().now().nanoseconds / 1e9
                if(self.stack_flag):
                    self.publish_cmd(
                        0, 0, 0,
                        self.stack_drop_base,
                        self.stack_drop_elbow
                    )
                else:
                    self.publish_cmd(
                        0, 0, 0,
                        # self.default_base,
                        # self.default_elbow
                        0.0,
                        90.0
                    )
                return

            now = self.get_clock().now().nanoseconds / 1e9

            if now - self.pickup_start_time < self.PICKUP_DELAY:
                # keep holding position
                if(self.stack_flag):
                    self.publish_cmd(
                        0, 0, 0,
                        self.stack_drop_base,
                        self.stack_drop_elbow
                    )
                else:
                    self.publish_cmd(
                        0, 0, 0,
                        # self.default_base,
                        # self.default_elbow
                        0.0,
                        90.0
                    )
                return
            self.call_attach(False)
            self.notify_planner()
            print("Notified the planner")
            # ===== A1: CLEAR TASK STATE (REQUIRED) =====
            self.crate_pose = None
            self.tx = None
            self.ty = None
            self.path_crate = []
            self.path_drop = []
            self.path_return = []

            self.wp_idx_crate = 0
            self.wp_idx_drop = 0
            self.wp_idx_return = 0
            # =========================================
            # delay done → move on
            self.pickup_start_time = None
            self.pid_x.reset()
            self.pid_y.reset()
            
            self.crates_delivered += 1

            self.crate_pose = None
            self.path_crate = []
            self.path_drop = []
            self.wp_idx_crate = 0
            self.wp_idx_drop = 0

            self.pid_x.reset()
            self.pid_y.reset()
            self.pickup_start_time = None

            if self.crates_delivered < self.MAX_CRATES:
                self.notify_planner()
                # print("Notified the planner")
                self.notify_planner()
                # print("Notified the planner")
                self.state = "PATH_TO_CRATE"
                self.STOP_RADIUS = 70.0
                if(self.crates_delivered == 2):
                    # print("Stacking mode activated")
                    self.stack_flag = True
                    print("STACK FLAG: ", self.stack_flag)
                print("PATH_TO_CRATE_STARTED")
            else:
                # self.notify_planner()
                # print("Notified the planner")
                self.stack_flag = False
                print("STACK FLAG: ", self.stack_flag)
                self.state = "PATH_TO_HOME"
                print("PATH_TO_HOME_STARTED")

            


        # -------- PLAN RETURN HOME (NEW) --------
        elif self.state == "PATH_TO_HOME":
            if len(self.path_return) == 0:
                return
            self.wp_idx_return = 0
            self.state = "MOVE_TO_START"
            self.get_logger().info("PATH_TO_HOME → MOVE_TO_START")
            return

            # # ---------- build obstacle grid (crate is attached, ignore all) ----------
            # grid = self.planner.build_grid(
            #     self.last_crates,
            #     ignore_id=None,
            #     bot_id = self.BOT_ID,
            #     allow_d1=True
            # )

            # # ---------- plan A* path to home ----------
            # self.path = self.planner.plan(
            #     start=(self.bot_pose.x, self.bot_pose.y),
            #     goal=(self.start_x, self.start_y),
            #     grid=grid
            # )

            # # ---------- publish debug info ----------
            # self.publish_planner_debug(
            #     grid=grid,
            #     path=self.path,
            #     start=(self.bot_pose.x, self.bot_pose.y),
            #     goal=(self.start_x, self.start_y)
            # )

            # # ---------- safety check ----------
            # if not self.path:
            #     self.get_logger().error("❌ A* failed to find path to home")
            #     self.stop_robot()
            #     return

            # self.wp_idx = 0
            # self.state = "MOVE_TO_START"
            # self.get_logger().info("PATH_TO_HOME STATE DONE")
            # self.get_logger().info("MOVE_TO_START STATE START")


        elif self.state == "MOVE_TO_START":
            # self.call_attach(False)
            # wx, wy = self.path[self.wp_idx]
            # if self.drive_to(wx, wy, self.default_base, self.default_elbow):
            #     self.wp_idx += 1
            #     if self.wp_idx >= len(self.path):
            #         self.state = "STOP_AT_START"
            #         self.get_logger().info("MOVE_TO_START STATE DONE")
            #         self.get_logger().info("STOP_AT_START STATE START")

            if self.wp_idx_return >= len(self.path_return):
                self.stop_robot()
                self.state = "STOP_AT_START"
                #self.tx, self.ty = self.crate_pose.x, self.crate_pose.y
                self.notify_planner()
                print("Notified the planner")
                self.get_logger().info("MOVE_TO_START STATE DONE")
                self.get_logger().info("STOP_AT_START STATE START")
                return

            tx, ty = self.path_return[self.wp_idx_return]
            final = (self.wp_idx_return == len(self.path_return) - 1)

            if self.drive_to(tx, ty, final):
                self.wp_idx_return += 1
                

        elif self.state == "STOP_AT_START":
            self.stop_robot()
            self.state = "ALIGN_AT_START"
            self.get_logger().info("STOP_AT_START STATE DONE")
            self.get_logger().info("ALIGN_AT_START STATE START")
            return

        elif self.state == "ALIGN_AT_START":
            byaw = math.radians(self.bot_pose.w)

            # target yaw = 0 rad
            err = wrap_angle(0.0 - byaw)

            if abs(err) < math.radians(12.0):  # ~2 deg tolerance
                self.publish_cmd(0.0, 0.0, 0.0,
                                self.default_base,
                                self.default_elbow)
                self.state = "COMPLETE"
                self.get_logger().info("ALIGN_AT_START DONE")
                return

            K = 20.0
            rot = K * err
            print(err)

            # minimum angular speed
            MIN_ROT = 8.0
            if abs(rot) < MIN_ROT:
                rot = math.copysign(MIN_ROT, rot)
                

            self.publish_cmd(rot, rot, rot,
                            self.default_base,
                            self.default_elbow)


        elif self.state == "COMPLETE":
            self.stop_robot()
            self.get_logger().info("COMPLETED")



        


def main():
    rclpy.init()
    node = HolonomicController()
    rclpy.spin(node)
    node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()