# Grid Path Planner (ROS2)

Centralized multi-robot path planner for an arena-based crate pickup and drop system.
Implements task allocation, grid-based navigation, collision avoidance, and structured drop-zone placement.

---

# Overview

This planner controls multiple mobile robots operating inside a **2.4384 m × 2.4384 m arena**.
Robots must:

1. Navigate to crates detected by perception
2. Align perpendicular to the crate
3. Pick up the crate
4. Navigate to the correct drop zone
5. Place crates in structured stacking patterns
6. Return to home positions

The planner operates entirely in **grid space** while perception provides **continuous world coordinates**.

---

# Core Capabilities

### Multi-Robot Path Planning

Uses **A*** search on a discrete grid to generate collision-free navigation paths.

### Task Allocation

Crates are assigned to bots based on nearest distance.

### Spatial Reservation

Paths reserve surrounding grid cells to enforce minimum robot separation.

### Collision Resolution

When robots approach too closely:

* higher priority robot continues
* lower priority robot computes a **backoff waypoint**
* planner resumes after clearance

### Structured Drop Placement

Drop zones enforce deterministic placement rules:

1. First crate → free base cell
2. Second crate → adjacent base cell
3. Third crate → stacked on base

### Entry Alignment

Robots do not directly enter drop cells.

Instead they:

```
A* navigation → pink entry strip → controller drives straight into drop cell
```

This isolates precise placement from navigation noise.

---

# System Architecture

```
Camera → Homography → World coordinates
                     ↓
                Grid planner
                     ↓
           A* path planning
                     ↓
       Path reservation / collision logic
                     ↓
            Path published to controller
```

Arena discretization:

```
Arena size      : 2438.4 mm
Grid resolution : 96 × 96
Cell size       : ≈25.4 mm
```

---

# Planner State Machine

Each robot operates under a finite state machine.

```
IDLE
  ↓
TO_CRATE
  ↓
TO_PINK (drop entry strip)
  ↓
TO_HOME
  ↓
IDLE
```

Transitions are triggered by controller notifications on:

```
/planner/path_done
```

---

# ROS2 Topics

## Subscriptions

### `/camera/image_raw`

```
sensor_msgs/Image
```

Raw camera frame used for arena visualization.

---

### `/arena/homography`

```
Float64MultiArray (9 elements)
```

Homography matrix mapping camera image → arena world frame.

---

### `/bot_pose`

```
hb_interfaces/Poses2D
```

World-frame pose estimates for all robots.

Fields:

```
id
x
y
yaw
```

---

### `/crate_pose`

```
hb_interfaces/Poses2D
```

Detected crate poses.

Fields:

```
id
x
y
yaw
```

---

### `/planner/path_done`

```
std_msgs/Int32
```

Controller notification that a robot completed its current path.

---

# Publications

## Navigation Paths

Paths are published as world-coordinate waypoint sequences.

```
Float64MultiArray
[x1,y1,x2,y2,x3,y3,...]
```

Topics:

```
/planner/path_to_crate_bot0
/planner/path_to_crate_bot2
/planner/path_to_crate_bot4

/planner/path_to_pink_bot0
/planner/path_to_pink_bot2
/planner/path_to_pink_bot4

/planner/path_to_home_bot0
/planner/path_to_home_bot2
/planner/path_to_home_bot4
```

---

## Pickup Target

```
/planner/pickup_target
Float64MultiArray
[x, y, yaw]
```

Defines the exact docking pose for crate pickup.

Robot approaches **perpendicular to crate orientation**.

---

## Backoff Target

```
/planner/backoff_target
Float64MultiArray
[x, y, yaw]
```

Published when collision resolution requires a robot to retreat.

---

## Assigned Crate IDs

Latched topics informing controllers which crate they are responsible for.

```
/planner/crate_id_bot0
/planner/crate_id_bot2
/planner/crate_id_bot4
```

---

## Stack Flag

Signals when a crate must be stacked on top of an existing base crate.

```
/planner/stack_flag_bot0
/planner/stack_flag_bot2
/planner/stack_flag_bot4
```

---

# Drop Zone Layout

Three drop zones exist:

| Zone | Color | Function        |
| ---- | ----- | --------------- |
| D1   | Red   | base + stacking |
| D2   | Green | base + stacking |
| D3   | Blue  | base + stacking |

Each zone contains a structured grid of drop cells.

Example:

```
D1 : 4 × 6 grid
D2 : 4 × 4 grid
D3 : 4 × 4 grid
```

Cells are generated deterministically inside the zone boundaries.

---

# Collision Handling

Robots maintain minimum separation:

```
COLLISION_RADIUS = 220 mm
```

If violated:

1. pair priority evaluated
2. lower priority robot computes retreat waypoint
3. backoff command issued
4. planning resumes after clearance

Backoff distance:

```
BACKOFF_MM = 100
```

---

# Path Reservation

Each published path reserves a corridor:

```
PATH_SEPARATION_MM = 250
```

Grid cells surrounding the path are marked occupied to prevent other robots planning through the same space.

---

# Visualization

The planner provides a debug window showing:

* arena grid
* drop zone boundaries
* entry strips
* robot positions
* planned paths
* crate local grids

Display resolution:

```
900 × 900 pixels
```

---

# Execution

Run node:

```
ros2 run <package_name> grid_planner
```

Node name:

```
grid_planner
```

Main loop:

```
rclpy.spin_once()
planner.spin_once()
```

---

# Key Parameters

| Parameter           | Value  | Description            |
| ------------------- | ------ | ---------------------- |
| ARENA_MM            | 2438.4 | arena size             |
| GRID_N              | 96     | grid resolution        |
| CELL_MM             | 25.4   | cell size              |
| COLLISION_RADIUS_MM | 220    | collision threshold    |
| PATH_SEPARATION_MM  | 250    | path reservation width |
| BACKOFF_MM          | 100    | retreat distance       |

---

# Dependencies

```
rclpy
opencv-python
numpy
cv_bridge
ROS2 message packages
```

Custom interface:

```
hb_interfaces/Poses2D
```

---

# Limitations

* Planner uses spatial reservations only (no time dimension).
* Greedy nearest-distance crate assignment.
* Entry corridor resource locking currently disabled.
* Performance optimized for ≤3 robots.

---

# License

Project intended for research / robotics competition use.
