#!/usr/bin/env python3
"""
Grid Path Planner with Perpendicular Pickup Target

- Uses world-frame poses from perception
- Full 24x24 grid covering entire arena
- A* path planning in grid space
- Computes precise pickup (x, y, yaw) perpendicular to crate
- Publishes pickup target for controller
"""
import cv2
import heapq
import math
from matplotlib import scale
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from hb_interfaces.msg import Poses2D
from std_msgs.msg import Int32

# ================= CONSTANTS =================
ARENA_MM = 2438.4
GRID_N = 120 #24
CELL_MM = ARENA_MM / GRID_N

LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
CRATE_SIZE_MM = 50.0
CRATE_GRID_CELLS = 1
CRATE_GRID_SIZE_MM = CRATE_SIZE_MM * CRATE_GRID_CELLS


PICKUP_OFFSET_MM = 1.0  # <<< TUNABLE

# DROP_ZONES = {
#     0: (200, 2200),
#     2: (1200, 2200),
#     4: (2200, 2200)
# }
# crate_id % 3 → drop zone
# 0 = RED, 1 = GREEN, 2 = BLUE

# ================= DROP ZONE GRID =================
DROP_CELL_MM = 65.0
DROP_INSET_MM = 20.0

# Drop zone outer bounds (WORLD MM)
# Replace these with the correct values from rulebook / arena
DROP_ZONES = {
    0: {  # D1 RED
        "xmin": 1020.0,
        "xmax": 1410.0,
        "ymin": 1075.0,
        "ymax": 1355.0,
    },
    1: {  # D2 GREEN
        "xmin": 675.0,
        "xmax": 965.0,
        "ymin": 1920.0,
        "ymax": 2115.0,
    },
    2: {  # D3 BLUE
        "xmin": 1470.0,
        "xmax": 1762.0,
        "ymin": 1920.0,
        "ymax": 2115.0,
    }
}

RESTRICTED_ZONE = set()
for fx in range(700, 950):
    for fy in range(50, 250):
        RESTRICTED_ZONE.add((
            int(fx / CELL_MM),
            int(fy / CELL_MM)
))

RESTRICTED_ZONE_BOT_2 = set()
for fx in range(0, 1500):
    for fy in range(1015, 1700):
        RESTRICTED_ZONE_BOT_2.add((
            int(fx / CELL_MM),
            int(fy / CELL_MM)
))
        
        



HOME_ZONES = {
    0: (1250, 140),
    2: (1610, 140),
    4: (884, 140)
}


BOT_COLORS = {
    0: (0, 255, 255),
    2: (0, 255, 0),
    4: (255, 0, 255),
}

WORLD_CORNERS_MM = np.array([
    [0, 0],
    [ARENA_MM, 0],
    [ARENA_MM, ARENA_MM],
    [0, ARENA_MM]
], dtype=np.float32)


# ---------- FIXED DROP TARGETS FOR BOT 0 ----------
BOT2_DROP_POINTS = [
    (900.0, 1900.0),
    (1570.0, 1885.0)    
]
''' (900.0, 1900.0),
    (1570.0, 1885.0)'''
# ---------- FIXED DROP TARGETS FOR BOT 0 ----------
BOT0_DROP_POINTS = [
    (1015.0, 1200.0),
    (1570.0, 1950.0) 
    
]

# ---------- FIXED DROP TARGETS FOR BOT 0 ----------
BOT4_DROP_POINTS = [
    (1005.0, 1257.0),
    (1005.0, 1166.0)
]

CRATE_SEQUENCE = {
    4: [12,21],
    0: [30,],
    2: [16,20]    # bot 0 handles crates in this order
}


# ================= A* =================
def astar(start, goal, forbidden, bot = None, return_path_flag = False):
    pq = [(0, start)]
    came = {}
    cost = {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            break

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            nxt = (nx, ny)

            if not (0 <= nx < GRID_N and 0 <= ny < GRID_N):
                continue
            if return_path_flag:
                if nxt in forbidden:
                    continue
                if nxt in RESTRICTED_ZONE:
                    continue
            if bot == 2:
                if nxt in RESTRICTED_ZONE_BOT_2:
                    continue
            # if nxt in forbidden:
            #     continue

            nc = cost[cur] + 1
            if nxt not in cost or nc < cost[nxt]:
                cost[nxt] = nc
                pr = nc + abs(nx-goal[0]) + abs(ny-goal[1])
                heapq.heappush(pq, (pr, nxt))
                came[nxt] = cur

    if goal not in came:
        return []

    path = [goal]
    while path[-1] != start:
        path.append(came[path[-1]])
    path.reverse()
    return path



# ================= NODE =================
class GridPlanner(Node):
    def __init__(self):
        super().__init__("grid_planner")
        self.drop_used = {0: set(), 1: set(), 2: set()}# Drop zone used cells
        
        self.occupied = set()    # (gx,gy) cells already reserved by previous bots
        
        self.planned = False

        # planner state
        self.bot_phase = {0: "IDLE", 2: "IDLE", 4: "IDLE"}
        self.assignments = {}    # bot_id -> crate_id
        self.active_paths = {}   # bot_id → [(gx,gy)]
        self.bot_phase = {}     # bot_id → "IDLE" | "TO_CRATE" | "TO_DROP" | "TO_HOME"
        self.carrying = set()   # bots holding a crate
        # self.unassigned_crates = set()
        for b in [0, 2, 4]:
            self.bot_phase[b] = "IDLE"

        # bot_id -> [(gx,gy)]
        self.path_published = set()

        self.bridge = CvBridge()
        self.frame = None
        self.H = None
        
        self.bot0_drop_index = 0            #################
        self.bot2_drop_index = 0
        self.bot4_drop_index = 0
        self.crate_seq_index = {0: 0, 2: 0, 4: 0}
        self.crate_assign_counter = {0: 0, 2: 0, 4: 0}



        self.bots = {}
        self.crates = {}

        # ---------- ROS ----------
        self.create_subscription(Image, "/camera/image_raw", self.image_cb, 10)
        self.create_subscription(Float64MultiArray, "/arena/homography", self.h_cb, 10)
        self.create_subscription(Poses2D, "/bot_pose", self.bot_cb, 10)
        self.create_subscription(Poses2D, "/crate_pose", self.crate_cb, 10)

        # single subscription to controller notifications (no duplicates)
        self.create_subscription(Int32, "/planner/path_done", self.done_cb, 10)
        
        self.pickup_pub = self.create_publisher(Float64MultiArray,"/planner/pickup_target",LATCHED_QOS)

        self.path_pubs = {
            "to_crate": {
                0: self.create_publisher(Float64MultiArray, "/planner/path_to_crate_bot0", LATCHED_QOS),
                2: self.create_publisher(Float64MultiArray, "/planner/path_to_crate_bot2", LATCHED_QOS),
                4: self.create_publisher(Float64MultiArray, "/planner/path_to_crate_bot4", LATCHED_QOS),
            },
            "to_drop": {
                0: self.create_publisher(Float64MultiArray, "/planner/path_to_drop_bot0", LATCHED_QOS),
                2: self.create_publisher(Float64MultiArray, "/planner/path_to_drop_bot2", LATCHED_QOS),
                4: self.create_publisher(Float64MultiArray, "/planner/path_to_drop_bot4", LATCHED_QOS),
            },
            "to_home": {
                0: self.create_publisher(Float64MultiArray, "/planner/path_to_home_bot0", LATCHED_QOS),
                2: self.create_publisher(Float64MultiArray, "/planner/path_to_home_bot2", LATCHED_QOS),
                4: self.create_publisher(Float64MultiArray, "/planner/path_to_home_bot4", LATCHED_QOS),
            }
        }

        self.crate_id_pubs = {
            0: self.create_publisher(Int32, "/planner/crate_id_bot0", LATCHED_QOS),
            2: self.create_publisher(Int32, "/planner/crate_id_bot2", LATCHED_QOS),
            4: self.create_publisher(Int32, "/planner/crate_id_bot4", LATCHED_QOS),
        }

        
        cv2.namedWindow("Grid Path Planner", cv2.WINDOW_NORMAL)
    def draw_drop_zone_grid(self, disp, color, scale):
        z = DROP_ZONES[color]

        xmin = z["xmin"] + DROP_INSET_MM
        xmax = z["xmax"] - DROP_INSET_MM
        ymin = z["ymin"] + DROP_INSET_MM
        ymax = z["ymax"] - DROP_INSET_MM

        col = (0,0,255) if color == 0 else (0,255,0) if color == 1 else (255,0,0)

        # vertical lines
        x = xmin
        while x <= xmax:
            px = int(x * scale)
            py1 = int(ymin * scale)
            py2 = int(ymax * scale)
            cv2.line(disp, (px, py1), (px, py2), col, 2)
            x += DROP_CELL_MM

        # horizontal lines
        y = ymin
        while y <= ymax:
            py = int(y * scale)
            px1 = int(xmin * scale)
            px2 = int(xmax * scale)
            cv2.line(disp, (px1, py), (px2, py), col, 2)
            y += DROP_CELL_MM

       
    def crate_edge_goal_cells(self, cx_mm, cy_mm, goal_offset_x = -3, goal_offset_y = 0):
        """
        Returns the 4 edge-center grid cells of the crate's 3x3 grid.
        Crate is centered at (cx_mm, cy_mm).
        """
        cxg, cyg = self.world_to_grid(cx_mm, cy_mm)
        print(f"crate center for bot: ", cxg, " ", cyg, " | after, offset, goal: ", cxg+ goal_offset_x, " ", cyg + goal_offset_y)

        # 3x3 grid → offsets
        # center = (0,0)
        # edge centers = up, down, left, right
        return [
            # (cxg,     cyg - 5),  # top edge
            (cxg + goal_offset_x, cyg + goal_offset_y),      # right edge
            # (cxg,     cyg + 5),  # bottom edge
            # (cxg, cyg),      # left edge
        ]
                
    def draw_crate_local_grid(self, disp, cx, cy, scale_px_per_mm):
        """
        Draws a 3x3 grid centered on the crate.
        Each cell = 50mm x 50mm
        Grid is visual-only.
        """
        half = CRATE_GRID_SIZE_MM / 2.0
        start_x = cx - half
        start_y = cy - half

        for i in range(CRATE_GRID_CELLS + 1):
            wx = start_x + i * CRATE_SIZE_MM
            px = int(wx * scale_px_per_mm)
            py1 = int(start_y * scale_px_per_mm)
            py2 = int((start_y + CRATE_GRID_SIZE_MM) * scale_px_per_mm)

            cv2.line(disp, (px, py1), (px, py2), (0, 0, 0), 2)

            wy = start_y + i * CRATE_SIZE_MM
            py = int(wy * scale_px_per_mm)
            px1 = int(start_x * scale_px_per_mm)
            px2 = int((start_x + CRATE_GRID_SIZE_MM) * scale_px_per_mm)

            cv2.line(disp, (px1, py), (px2, py), (0, 0, 0), 2)

    def inflate(self, cell, r=10):
        x, y = cell
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_N and 0 <= ny < GRID_N:
                    self.occupied.add((nx, ny))


    
    def compute_and_publish(self, bot):
        # safety: ensure bot pose known
        if bot not in self.bots:
            self.get_logger().warn(f"compute_and_publish: unknown bot {bot}")
            return

        # reset occupied and reserve cells for other active paths
        self.occupied = set()
        for b, path in self.active_paths.items():
            if b != bot:
                for c in path:
                    self.inflate(c, 10)

        bx, by = self.bots[bot]
        start = self.world_to_grid(bx, by)

        phase = self.bot_phase.get(bot, "IDLE")

        if phase == "TO_CRATE":
            # require assignment
            if bot not in self.assignments:
                self.get_logger().error(f"compute_and_publish: bot {bot} has no crate assignment")
                return
            crate = self.assignments[bot]
            print(f"BOT {bot} got crate {crate}")
            if crate not in self.crates:
                self.get_logger().error(f"compute_and_publish: crate {crate} data missing")
                return
            cx, cy, yaw = self.crates[crate]
            if(crate == 12):
                goal_offset_x = 4
                goal_offset_y = 0
                print(f"Offset for bot {bot} : ", goal_offset_x, " ", goal_offset_y)
            elif (crate == 30):
                goal_offset_x = -3
                goal_offset_y = 1
            elif (crate == 14):
                goal_offset_x = 4
                goal_offset_y = -2
            elif (crate == 21):
                goal_offset_x = 4
                goal_offset_y = 2
            elif (crate == 16):
                goal_offset_x = -3
                goal_offset_y = 0
            elif (crate == 20):
                goal_offset_x = -4
                goal_offset_y = 0
            edge_cells = self.crate_edge_goal_cells(cx, cy, goal_offset_x, goal_offset_y)
            goal = None
            best_len = float("inf")

            for cell in edge_cells:
                if cell in self.occupied:
                    continue
                candidate = astar(start, cell, self.occupied)
                print(candidate)
                if candidate and len(candidate) < best_len:
                    best_len = len(candidate)
                    goal = cell
            print("Goal is: ", goal)
            if goal is None:
                self.get_logger().error(f"NO EDGE GOAL for bot {bot}")
                return
    
    
            topic = "to_crate"

        elif phase == "TO_DROP":
            if bot not in self.assignments:
                self.get_logger().error(f"compute_and_publish: bot {bot} has no crate assignment for TO_DROP")
                return
            crate = self.assignments[bot]
            goal = self.pick_drop_goal(crate, bot)
            if goal is None:
                self.get_logger().error(f"compute_and_publish: no drop goal available for crate {crate}")
                return
            topic = "to_drop"

        elif phase == "TO_HOME":
            print("PHASE: TO_HOME")
            hx, hy = HOME_ZONES[bot]
            goal = self.world_to_grid(hx, hy)
            print("Home: ", goal, " from ", (hx, hy))
            topic = "to_home"

        else:
            # nothing to do
            return

        # -------- RUN A* --------
        
        if phase == "TO_HOME" and bot != 4:
            path = astar(start, goal, self.occupied, bot, return_path_flag=True)
            print("Return_flag is True for TO_HOME")
        else:
            path = astar(start, goal, self.occupied, bot, return_path_flag=False)
            print("Return_flag is False for TO_CRATE/TO_DROP")
        if not path:
            self.get_logger().error(f"NO PATH for bot {bot} (start={start}, goal={goal})")
            return
        print(path)

        # while not path:
        #     self.get_logger().error(f"NO PATH for bot {bot} (start={start}, goal={goal}), Retrying")
        #     path = astar(start, goal, self.occupied)
        # self.retry_bot0 = True      #########################
        # -------- STORE + RESERVE --------
        self.active_paths[bot] = path
        for c in path:
            self.inflate(c, 10)

        # -------- PUBLISH --------
        msg = Float64MultiArray()
        for gx, gy in path:
            msg.data.extend([
                (gx + 0.5) * CELL_MM,
                (gy + 0.5) * CELL_MM
            ])

        self.path_pubs[topic][bot].publish(msg)

        if phase == "TO_CRATE":
            crate_msg = Int32()
            crate_msg.data = crate
            self.crate_id_pubs[bot].publish(crate_msg)
        

    def pick_drop_goal(self, crate_id, bot_id = None):
        # ---------- SPECIAL RULE: BOT 0 FIXED DROPS ----------
        if bot_id == 0:
            if self.bot0_drop_index >= len(BOT0_DROP_POINTS):
                return None  # no more drop points

            wx, wy = BOT0_DROP_POINTS[self.bot0_drop_index]
            self.bot0_drop_index += 1

            return self.world_to_grid(wx, wy)
        
        # ---------- SPECIAL RULE: BOT 0 FIXED DROPS ----------
        if bot_id == 2:
            if self.bot2_drop_index >= len(BOT2_DROP_POINTS):
                return None  # no more drop points

            wx, wy = BOT2_DROP_POINTS[self.bot2_drop_index]
            self.bot2_drop_index += 1

            return self.world_to_grid(wx, wy)
        
        # ---------- SPECIAL RULE: BOT 0 FIXED DROPS ----------
        if bot_id == 4:
            if self.bot4_drop_index >= len(BOT4_DROP_POINTS):
                return None  # no more drop points

            wx, wy = BOT4_DROP_POINTS[self.bot4_drop_index]
            self.bot4_drop_index += 1

            return self.world_to_grid(wx, wy)

    # ---------- NORMAL LOGIC FOR OTHER BOTS ----------



        color = crate_id % 3
        drop_cells = self.build_drop_grid(color)

        for (ix, iy, wx, wy) in drop_cells:
            if (ix, iy) in self.drop_used[color]:
                continue

            gx, gy = self.world_to_grid(wx, wy)
            cell = (gx, gy)

            if cell in self.occupied:
                continue

            for path in self.active_paths.values():
                if cell in path:
                    break
            else:
                # reserve immediately
                self.drop_used[color].add((ix, iy))
                return cell

        return None

    def build_drop_grid(self, color):
        z = DROP_ZONES[color]

        xmin = z["xmin"] + DROP_INSET_MM
        xmax = z["xmax"] - DROP_INSET_MM
        ymin = z["ymin"] + DROP_INSET_MM
        ymax = z["ymax"] - DROP_INSET_MM

        cells = []
        ix = 0
        x = xmin
        while x + DROP_CELL_MM <= xmax:
            iy = 0
            y = ymin
            while y + DROP_CELL_MM <= ymax:
                cells.append((ix, iy, x + DROP_CELL_MM/2, y + DROP_CELL_MM/2))
                iy += 1
                y += DROP_CELL_MM
            ix += 1
            x += DROP_CELL_MM

        return cells


    
    # def assign_crates(self):
    #     if self.assignments:
    #         return

    #     if not self.bots or not self.crates:
    #         return
        
    #     unused_bots   = set(self.bots.keys()) - set(self.assignments.keys())
    #     unused_crates = set(self.crates.keys()) - set(self.assignments.values())

    #     while unused_bots and unused_crates:
    #         best = None
    #         best_dist = float("inf")

    #         for b in unused_bots:
    #             bx, by = self.bots[b]
    #             for c in unused_crates:
    #                 cx, cy, _ = self.crates[c]
    #                 d = (bx - cx)**2 + (by - cy)**2
    #                 if d < best_dist:
    #                     best_dist = d
    #                     best = (b, c)

    #         b, c = best
    #         self.assignments[b] = c
    #         unused_bots.remove(b)
    #         unused_crates.remove(c)

    #         self.get_logger().info(
    #             f"Assigned crate {c} → bot {b} (distance={math.sqrt(best_dist):.1f}mm)"
    #         )

    def assign_crates(self):
        for bot, seq in CRATE_SEQUENCE.items():
            if bot in self.assignments:
                continue
            idx = self.crate_seq_index[bot]
            if idx >= len(seq):
                continue
            crate = seq[idx]
            if crate in self.crates:
                self.assignments[bot] = crate


    # ---------- Callbacks ----------
    def image_cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def h_cb(self, msg):
        if len(msg.data) == 9:
            self.H = np.array(msg.data, dtype=np.float32).reshape(3, 3)

    def bot_cb(self, msg):
        for p in msg.poses:
            self.bots[p.id] = (p.x, p.y)


    def crate_cb(self, msg):
        for p in msg.poses:
            self.crates[p.id] = (p.x, p.y, p.w)


    # ---------- Helpers ----------
    def world_to_grid(self, x, y):
        return int(x / CELL_MM), int(y / CELL_MM)

    def compute_pickup_target(self, crate):
        cx, cy, yaw_deg = crate
        theta = math.radians(yaw_deg)

        # Crate outward normal
        nx = math.sin(theta)
        ny = -math.cos(theta)
        tx = cx + nx * PICKUP_OFFSET_MM
        ty = cy + ny * PICKUP_OFFSET_MM
        desired_yaw = (theta - math.pi / 2.0)
        desired_yaw = math.atan2(math.sin(desired_yaw), math.cos(desired_yaw))

        return tx, ty, desired_yaw
    
    def start_planning(self):
        # self.unassigned_crates = set(self.crates.keys())
        self.assign_crates()
        for bot in self.assignments:
            self.plan_next(bot)

    # def done_cb(self, msg):
    #     bot = msg.data

    #     # capture old path (may be None)
    #     old_path = self.active_paths.get(bot)

    #     # advance phase and compute NEW path FIRST
    #     try:
    #         self.plan_next(bot)
    #     except Exception as e:
    #         self.get_logger().error(f"done_cb: plan_next failed for bot {bot}: {e}")
    #         return

    #     # now free cells from OLD path
    #     if old_path:
    #         for c in old_path:
    #             self.occupied.discard(c)

    #     # handle crate bookkeeping AFTER drop
    #     # if self.bot_phase.get(bot) == "TO_HOME":
    #     #     finished = self.assignments.pop(bot, None)
    #     #     if finished is not None:
    #     #         self.unassigned_crates.discard(finished)
        
    #     if self.bot_phase.get(bot) == "TO_HOME":
    #         finished = self.assignments.pop(bot, None)
    #         if finished is not None:
    #             self.crate_seq_index[bot] += 1
    #             self.assign_crates()
    
    def done_cb(self, msg):
        bot = msg.data

        old_path = self.active_paths.get(bot)

        phase = self.bot_phase.get(bot)

        # ----- PHASE COMPLETION LOGIC FIRST -----
        if phase == "TO_DROP":
            finished = self.assignments.pop(bot, None)
            if finished is not None:
                print(f"Bot {bot} completed drop of crate {finished}")
                self.crate_seq_index[bot] += 1
                self.assign_crates()

        # ----- ADVANCE STATE AFTER BOOKKEEPING -----
        try:
            self.plan_next(bot)
        except Exception as e:
            self.get_logger().error(f"done_cb: plan_next failed for bot {bot}: {e}")
            return

        # ----- FREE OLD PATH -----
        if old_path:
            for c in old_path:
                self.occupied.discard(c)
                self.active_paths.pop(bot, None)




        

    def plan_next(self, bot):
        phase = self.bot_phase.get(bot, "IDLE")

        if phase == "IDLE":
            self.bot_phase[bot] = "TO_CRATE"
            print(f"Bot {bot} going to TO_CRATE")

        elif phase == "TO_CRATE":
            self.crate_assign_counter[bot] += 1
            print("crate assign counter: ", self.crate_assign_counter[bot])
            self.bot_phase[bot] = "TO_DROP"
            print(f"Bot {bot} going to TO_DROP")

        elif phase == "TO_DROP":
            # if self.crate_seq_index .get(bot, 0) < len(CRATE_SEQUENCE.get(bot, [])):
            if self.crate_assign_counter[bot] <= 2 and bot == 4:
                print("crate sequence, crate seq :", len(CRATE_SEQUENCE.get(bot, [])), self.crate_seq_index.get(bot, 0))
                self.bot_phase[bot] = "TO_CRATE"
                print(f"Bot {bot} going to TO_CRATE")
            else:
                self.bot_phase[bot] = "TO_HOME"
                print(f"Bot {bot} going to TO_HOME")

        # elif phase == "TO_HOME":
        #     if self.crate_seq_index .get(bot, 0) < len(CRATE_SEQUENCE.get(bot, [])):
        #         self.bot_phase[bot] = "TO_CRATE"
        #         # self.assignments[bot] = self.unassigned_crates.pop()
        #         # self.bot_phase[bot] = "TO_CRATE"
        #     else:
        #         self.bot_phase[bot] = "IDLE"
        #         return
        elif phase == "TO_HOME":
            self.bot_phase[bot] = "IDLE"
            return



        self.compute_and_publish(bot)


    # ---------- Main ----------
    def spin_once(self):
        
        if not self.planned and self.bots and self.crates and self.H is not None:
            self.start_planning()
            self.planned = True
        
        # removed idle re-trigger to keep planner controlled by controller notifications

        if self.frame is None or self.H is None:
            cv2.waitKey(1)
            return

        warp = cv2.warpPerspective(
            self.frame, self.H,
            (int(ARENA_MM), int(ARENA_MM))
        )

        disp = cv2.resize(warp, (900, 900))
        # ---- Draw bot positions ----
        scale = 900 / ARENA_MM

        for bid, (bx, by) in self.bots.items():
            px = int((bx / ARENA_MM) * 900)
            py = int((by / ARENA_MM) * 900)

            cv2.circle(disp, (px, py), 10, (0, 0, 255), -1)

            label = f"bot{bid}: {bx:.0f},{by:.0f}"
            cv2.putText(
                disp,
                label,
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0,0,255),
                1
            )

        step = int(900 / GRID_N)

        # ----- Draw grid -----
        for i in range(GRID_N + 1):
            cv2.line(disp, (i*step,0), (i*step,900), (255,255,255), 1)
            cv2.line(disp, (0,i*step), (900,i*step), (255,255,255), 1)

        # ----- Draw paths + publish pickup target -----
        # ----- Draw + publish all phases -----
        for bot, path in self.active_paths.items():
            for i in range(1, len(path)):
                p1 = ((path[i-1][0]+0.5)*step, (path[i-1][1]+0.5)*step)
                p2 = ((path[i][0]+0.5)*step, (path[i][1]+0.5)*step)

                cv2.line(
                    disp,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    BOT_COLORS[bot],
                    4
                )
        for color in DROP_ZONES:
            self.draw_drop_zone_grid(disp, color, scale)

              
        # ----- Draw crate local 3x3 grids -----
        for cid, (cx, cy, _) in self.crates.items():
            self.draw_crate_local_grid(disp, cx, cy, scale)

        cv2.imshow("Grid Path Planner", disp)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = GridPlanner()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.spin_once()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()