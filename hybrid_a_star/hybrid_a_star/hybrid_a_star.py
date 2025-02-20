import math
import heapq
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# Car specs
WHEELBASE = 0.3302
MAX_STEER = math.radians(30)
STEER_ANGLES = [math.radians(a) for a in range(-30, 31, 10)]
SPEED = 1.0
DT = 0.1

# Hybrid A* discretization
YAW_RES_DEG = 5
YAW_GRID = 360 // YAW_RES_DEG

# Limits
GOAL_TOL = 0.3
MAX_ITERATIONS = 50000

class HybridAStarNode(Node):
    def __init__(self):
        super().__init__('hybrid_a_star_node')

        self.map_data = None
        self.map_resolution = 1.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_width = 0
        self.map_height = 0

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.goal_x = None
        self.goal_y = None

        # Subscription to map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST
            )
        )
        # Subscription to odom
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            1
        )
        # Subscription to goal
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            1
        )
        # Publisher for drive commands
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            1
        )
        # Publisher for visualizing the path in RViz
        self.path_pub = self.create_publisher(
            Path,
            '/hybrid_astar_path',
            1
        )

        # Timer to do planning/control
        self.timer = self.create_timer(0.1, self.timer_callback)

        # 2D BFS distance map for a better heuristic
        self.bfs_dist_map = None

    def map_callback(self, msg: OccupancyGrid):
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

        grid = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))
        self.map_data = grid

        if self.goal_x is not None:
            self.bfs_dist_map = self.compute_bfs_dist_map(self.goal_x, self.goal_y)

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.current_x = pos.x
        self.current_y = pos.y
        self.current_yaw = yaw

    def goal_callback(self, msg: PoseStamped):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        if self.map_data is not None:
            self.bfs_dist_map = self.compute_bfs_dist_map(self.goal_x, self.goal_y)

    def timer_callback(self):
        if self.map_data is None or self.bfs_dist_map is None or self.goal_x is None:
            return

        dist_to_goal = math.hypot(self.goal_x - self.current_x, self.goal_y - self.current_y)
        if dist_to_goal < GOAL_TOL:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)
            return

        path = self.hybrid_a_star(
            self.current_x, self.current_y, self.current_yaw,
            self.goal_x, self.goal_y
        )

        if path:
            # Publish the path so it can be visualized in RViz
            path_msg = self.convert_to_path_msg(path)
            self.path_pub.publish(path_msg)

            if len(path) > 1:
                target = path[1]
            else:
                target = path[0]

            cmd_steer = self.compute_steer(target)
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = SPEED
            drive_msg.drive.steering_angle = cmd_steer
            self.drive_pub.publish(drive_msg)
        else:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)
            self.get_logger().warn("Hybrid A* failed to find a path")

    def compute_steer(self, target):
        dx = target[0] - self.current_x
        dy = target[1] - self.current_y
        heading = math.atan2(dy, dx)
        alpha = heading - self.current_yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        steer = max(min(alpha, MAX_STEER), -MAX_STEER)
        return steer

    def hybrid_a_star(self, sx, sy, syaw, gx, gy):
        sx_idx, sy_idx, syaw_idx = self.world_to_grid(sx, sy, syaw)
        gx_idx, gy_idx = self.world_to_map(gx, gy)

        if not self.valid_cell(gx_idx, gy_idx):
            self.get_logger().warn("Goal is out of map bounds!")
            return None

        cost_3d = np.full((self.map_height, self.map_width, YAW_GRID), float('inf'), dtype=float)
        cost_3d[sy_idx, sx_idx, syaw_idx] = 0.0

        start_node = (0.0, (sx_idx, sy_idx, syaw_idx, None))
        open_list = [start_node]
        heapq.heapify(open_list)

        visited = set()

        iterations = 0
        while open_list:
            iterations += 1
            if iterations > MAX_ITERATIONS:
                self.get_logger().warn("Max iterations reached in Hybrid A*")
                return None

            current_cost, (cx, cy, cyaw_i, parent) = heapq.heappop(open_list)
            if (cx, cy, cyaw_i) in visited:
                continue
            visited.add((cx, cy, cyaw_i))

            wx, wy, _ = self.grid_to_world(cx, cy, cyaw_i)
            if math.hypot(wx - gx, wy - gy) < GOAL_TOL:
                return self.reconstruct_path(cx, cy, cyaw_i, parent)

            for steer in STEER_ANGLES:
                nx, ny, nyaw_i = self.predict(cx, cy, cyaw_i, steer)
                if nx < 0 or ny < 0 or nx >= self.map_width or ny >= self.map_height:
                    continue
                if self.collision_cell(nx, ny):
                    continue

                new_g = current_cost + DT
                bfs_cost = self.bfs_dist_map[ny, nx]
                if bfs_cost == float('inf'):
                    continue
                total_cost = new_g + bfs_cost

                if total_cost < cost_3d[ny, nx, nyaw_i]:
                    cost_3d[ny, nx, nyaw_i] = total_cost
                    heapq.heappush(open_list, (total_cost, (nx, ny, nyaw_i, (cx, cy, cyaw_i, parent))))

        return None

    def predict(self, x_i, y_i, yaw_i, steer):
        wx, wy, wyaw = self.grid_to_world(x_i, y_i, yaw_i)

        nx = wx + SPEED * math.cos(wyaw) * DT
        ny = wy + SPEED * math.sin(wyaw) * DT
        nyaw = wyaw + (SPEED / WHEELBASE) * math.tan(steer) * DT

        nx_i, ny_i, nyaw_i = self.world_to_grid(nx, ny, nyaw)
        return nx_i, ny_i, nyaw_i

    def reconstruct_path(self, x_i, y_i, yaw_i, parent):
        path = []
        node = (x_i, y_i, yaw_i, parent)
        while node is not None:
            px, py, pyaw_i, pparent = node
            wx, wy, wyaw = self.grid_to_world(px, py, pyaw_i)
            path.append((wx, wy, wyaw))
            node = pparent
        path.reverse()
        return path

    def world_to_grid(self, x, y, yaw):
        mx = int((x - self.map_origin_x) / self.map_resolution)
        my = int((y - self.map_origin_y) / self.map_resolution)
        yaw = (yaw + math.pi * 2) % (2 * math.pi)
        yaw_i = int(round((yaw / (2 * math.pi)) * YAW_GRID)) % YAW_GRID
        return mx, my, yaw_i

    def grid_to_world(self, mx, my, yaw_i):
        wx = mx * self.map_resolution + self.map_origin_x
        wy = my * self.map_resolution + self.map_origin_y
        yaw = (yaw_i / float(YAW_GRID)) * (2.0 * math.pi)
        return wx, wy, yaw

    def world_to_map(self, x, y):
        mx = int((x - self.map_origin_x) / self.map_resolution)
        my = int((y - self.map_origin_y) / self.map_resolution)
        return mx, my

    def valid_cell(self, mx, my):
        return (0 <= mx < self.map_width) and (0 <= my < self.map_height)

    def collision_cell(self, mx, my):
        if not self.valid_cell(mx, my):
            return True
        val = self.map_data[my, mx]
        return (val > 50 or val < 0)

    def compute_bfs_dist_map(self, gx, gy):
        dist_map = np.full((self.map_height, self.map_width), float('inf'), dtype=float)

        gm_x = int((gx - self.map_origin_x) / self.map_resolution)
        gm_y = int((gy - self.map_origin_y) / self.map_resolution)
        if not self.valid_cell(gm_x, gm_y) or self.collision_cell(gm_x, gm_y):
            self.get_logger().warn("Goal cell is invalid for BFS!")
            return dist_map

        from collections import deque
        q = deque()
        dist_map[gm_y, gm_x] = 0.0
        q.append((gm_x, gm_y))

        # 8-connected BFS
        directions = [(1,0), (-1,0), (0,1), (0,-1),
                      (1,1), (1,-1), (-1,1), (-1,-1)]

        while q:
            cx, cy = q.popleft()
            cur_dist = dist_map[cy, cx]
            for dx, dy in directions:
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    if not self.collision_cell(nx, ny):
                        nd = cur_dist + math.hypot(dx, dy) * self.map_resolution
                        if nd < dist_map[ny, nx]:
                            dist_map[ny, nx] = nd
                            q.append((nx, ny))

        return dist_map

    def convert_to_path_msg(self, path):
        """
        Convert a list of (x, y, yaw) into a nav_msgs/Path for RViz.
        """
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"  # change if needed

        for (px, py, pyaw) in path:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = px
            pose_stamped.pose.position.y = py

            # Convert yaw to quaternion
            q = quaternion_from_euler(0.0, 0.0, pyaw)
            pose_stamped.pose.orientation.x = q[0]
            pose_stamped.pose.orientation.y = q[1]
            pose_stamped.pose.orientation.z = q[2]
            pose_stamped.pose.orientation.w = q[3]

            path_msg.poses.append(pose_stamped)
        return path_msg

def main(args=None):
    rclpy.init(args=args)
    node = HybridAStarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
