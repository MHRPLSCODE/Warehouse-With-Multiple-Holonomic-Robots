# Multi-Robot Crate Handling System (ROS2)

Centralized perception-driven robotics system for autonomous crate collection, transport, and structured placement inside a constrained arena using multiple mobile robots.

The project integrates **vision-based localization, centralized path planning, robot controllers, and task coordination** using ROS2.

---

# System Objective

Multiple robots operate inside a square arena to:

1. Detect crates using overhead vision
2. Estimate crate and robot poses in a common world frame
3. Assign crates to robots
4. Plan collision-free paths
5. Perform perpendicular crate pickup
6. Deliver crates to designated drop zones
7. Execute structured stacking
8. Return robots to home positions

The architecture separates:

```text
Perception
Planning
Control
Visualization
```

This separation allows each subsystem to operate independently while communicating through ROS topics.

---

# System Architecture

```
                CAMERA
                   │
                   ▼
          Homography Estimator
                   │
                   ▼
           World Coordinate Frame
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   Robot Pose Node      Crate Pose Node
        │                     │
        └──────────┬──────────┘
                   ▼
             Grid Planner
                   │
        ┌──────────┴──────────┐
        ▼          ▼          ▼
    Bot0 Ctrl   Bot2 Ctrl   Bot4 Ctrl
        │          │          │
        ▼          ▼          ▼
      Robot       Robot      Robot
```

The **planner node acts as the centralized decision system**, while controllers execute trajectories locally.

---

# Arena Model

Arena size:

```
2438.4 mm × 2438.4 mm
```

Grid representation:

```
96 × 96 grid
```

Cell resolution:

```
≈25.4 mm
```

The grid is used only for path planning.
Robot motion remains continuous in world coordinates.

---

# Robots

Three robots operate simultaneously.

| Robot ID | Role         |
| -------- | ------------ |
| 0        | mobile robot |
| 2        | mobile robot |
| 4        | mobile robot |

Each robot has:

* differential or omni drive base
* crate pickup mechanism
* onboard controller node

---

# Major Components

## 1. Vision System

Responsible for detecting robot and crate poses.

Outputs:

```
/bot_pose
/crate_pose
```

Coordinate system is the **arena world frame (mm)**.

The world frame is produced by applying a **homography transform** from the camera image.

---

## 2. Homography Estimation

Computes transformation:

```
camera pixels → arena world coordinates
```

Published on:

```
/arena/homography
```

Used by the planner for visualization and coordinate conversion.

---

## 3. Grid Planner

Central planning node responsible for:

* task allocation
* path planning
* collision avoidance
* drop zone placement
* robot state machine

Planner uses:

```
A* search on discrete grid
```

while robots execute waypoints in world coordinates.

Planner outputs paths through ROS topics.

---

## 4. Robot Controllers

Each robot has a controller node responsible for:

* receiving waypoint paths
* following path
* performing crate pickup
* performing crate drop
* reporting path completion

Controller publishes:

```
/planner/path_done
```

when a path is finished.

---

## 5. Drop Zone Manager

Drop zones enforce placement rules.

Each zone:

```
structured grid
```

Crate placement logic:

```
1st crate → base cell
2nd crate → adjacent base cell
3rd crate → stacked
```

Zones:

| Zone | Color | Grid |
| ---- | ----- | ---- |
| D1   | Red   | 4×6  |
| D2   | Green | 4×4  |
| D3   | Blue  | 4×4  |

---

# Robot State Machine

Each robot follows this planner-driven state machine:

```
IDLE
  │
  ▼
TO_CRATE
  │
  ▼
TO_PINK_ENTRY
  │
  ▼
DROP_CRATE
  │
  ▼
TO_HOME
  │
  ▼
IDLE
```

State transitions occur when the controller signals path completion.

---

# Navigation Strategy

Robots do not drive directly into drop cells.

Instead they navigate in two stages:

```
A* path → entry alignment strip → straight insertion
```

This reduces path planning complexity near the drop zone.

---

# Collision Avoidance

The planner enforces spatial separation.

Minimum robot distance:

```
220 mm
```

If robots approach closer:

1. priority rule selects winner
2. losing robot computes retreat point
3. retreat command published
4. planning resumes once clearance achieved

---

# Task Allocation

Crates are assigned using a greedy nearest-distance rule.

```
distance(bot, crate)
```

Assignments update dynamically as crates are completed.

---

# ROS2 Topics

## Perception

```
/camera/image_raw
sensor_msgs/Image
```

```
/arena/homography
Float64MultiArray
```

```
/bot_pose
hb_interfaces/Poses2D
```

```
/crate_pose
hb_interfaces/Poses2D
```

---

## Planner Output

Paths:

```
/planner/path_to_crate_bot0
/planner/path_to_crate_bot2
/planner/path_to_crate_bot4
```

```
/planner/path_to_pink_bot0
/planner/path_to_pink_bot2
/planner/path_to_pink_bot4
```

```
/planner/path_to_home_bot0
/planner/path_to_home_bot2
/planner/path_to_home_bot4
```

Pickup target:

```
/planner/pickup_target
```

Backoff command:

```
/planner/backoff_target
```

Crate assignments:

```
/planner/crate_id_botX
```

Stacking flag:

```
/planner/stack_flag_botX
```

---

# Visualization

Planner includes a real-time debugging display.

Shows:

* arena grid
* robot positions
* crate locations
* drop zone grids
* entry alignment strips
* planned robot paths

Display resolution:

```
900 × 900
```

---

# Running the System

Launch order:

```
1. camera node
2. homography estimator
3. perception nodes
4. planner node
5. robot controller nodes
```

Example command:

```
ros2 run <package> grid_planner
```

---

# Dependencies

Required packages:

```
ROS2 (Humble or newer)
OpenCV
NumPy
cv_bridge
```

Custom message package:

```
hb_interfaces
```

Message used:

```
Poses2D
```

---

# Known Limitations

Planner uses **spatial reservation only**.

Limitations include:

* no time dimension in path planning
* greedy task allocation
* scaling limited to small robot counts
* entry corridor locking currently disabled

The system is optimized for **three robots**.

---

# Future Improvements

Potential improvements:

```
time-expanded multi-agent planning
CBS or SIPP planners
predictive collision avoidance
improved crate assignment optimization
entry corridor resource locking
```

---

# Project Purpose

Designed for:

```
multi-robot research
robotics competitions
autonomous warehouse experiments
```

The project demonstrates a complete robotics stack integrating **vision, planning, and control** in a structured arena environment.
