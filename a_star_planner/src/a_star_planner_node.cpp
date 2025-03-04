#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <queue>
#include <vector>
#include <memory>
#include <chrono>
#include <unordered_map>

using namespace std::chrono_literals;

// Custom Node for A* algorithm
struct AStarNode {
    int x, y;
    double g, h, f;
    AStarNode* parent;

    AStarNode(int x, int y, double g, double h, AStarNode* parent = nullptr)
        : x(x), y(y), g(g), h(h), f(g + h), parent(parent) {}

    bool operator<(const AStarNode& other) const {
        return f > other.f; // For priority queue (min-heap)
    }
};

struct MapPoint {
    double x, y;
};

class AStarPlanner : public rclcpp::Node {
public:
    AStarPlanner() : Node("a_star_planner") {
        declare_parameter("heuristic_type", "euclidean");
        declare_parameter("allow_diagonal", true);
        declare_parameter("lookahead_distance", 0.5);
        declare_parameter("speed", 1.5);
        declare_parameter("goal_tolerance", 0.3);

        heuristic_type_ = get_parameter("heuristic_type").as_string();
        allow_diagonal_ = get_parameter("allow_diagonal").as_bool();
        lookahead_distance_ = get_parameter("lookahead_distance").as_double();
        speed_ = get_parameter("speed").as_double();
        goal_tolerance_ = get_parameter("goal_tolerance").as_double();

        map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10, std::bind(&AStarPlanner::mapCallback, this, std::placeholders::_1));
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10, std::bind(&AStarPlanner::odomCallback, this, std::placeholders::_1));
        goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10, std::bind(&AStarPlanner::goalCallback, this, std::placeholders::_1));

        drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        path_pub_ = create_publisher<visualization_msgs::msg::Marker>("/path", 10);
        open_list_pub_ = create_publisher<visualization_msgs::msg::Marker>("/open_list", 10);
        closed_list_pub_ = create_publisher<visualization_msgs::msg::Marker>("/closed_list", 10);
        status_pub_ = create_publisher<visualization_msgs::msg::Marker>("/planning_status", 10);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        timer_ = create_wall_timer(50ms, std::bind(&AStarPlanner::controlLoop, this));
        planning_timer_ = create_wall_timer(100ms, std::bind(&AStarPlanner::publishStatus, this));

        RCLCPP_INFO(get_logger(), "A* Planner initialized");
    }

private:
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        RCLCPP_DEBUG(get_logger(), "Received new map");
        map_ = msg;
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        geometry_msgs::msg::PoseStamped odom_pose;
        odom_pose.header = msg->header;
        odom_pose.pose = msg->pose.pose;

        try {
            auto map_pose = tf_buffer_->transform(odom_pose, "map", tf2::durationFromSec(0.1));
            current_pose_ = toMapPoint(map_pose.pose);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(get_logger(), "Transform error: %s", ex.what());
        }
    }

    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        RCLCPP_INFO(get_logger(), "Received new goal");
        goal_reached_ = false;
        goal_point_ = toMapPoint(msg->pose);
        planPath();
    }

    void planPath() {
        if (!map_ || !current_pose_ || !goal_point_) {
            RCLCPP_WARN(get_logger(), "Missing data for planning");
            return;
        }

        auto start = worldToGrid(current_pose_->x, current_pose_->y);
        auto goal = worldToGrid(goal_point_->x, goal_point_->y);

        RCLCPP_INFO(get_logger(), "Planning path from (%d,%d) to (%d,%d)", 
                   start.first, start.second, goal.first, goal.second);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto path = aStar(start, goal);
        auto duration = std::chrono::high_resolution_clock::now() - start_time;

        RCLCPP_INFO(get_logger(), "Planning took %.2f ms. Path length: %zu", 
                   duration.count() / 1e6, path.size());

        if (!path.empty()) {
            RCLCPP_INFO(get_logger(), "Path found!");
            current_path_ = path;
            publishPath(path);
            goal_reached_ = false;
        } else {
            RCLCPP_ERROR(get_logger(), "No path found!");
            current_path_.clear();
        }

        publishSearchVisualization();
    }

    std::vector<MapPoint> aStar(std::pair<int, int> start, std::pair<int, int> goal) {
        std::priority_queue<AStarNode> open_list;
        std::vector<std::vector<bool>> closed_list(map_->info.height, 
                                                  std::vector<bool>(map_->info.width, false));
        std::vector<AStarNode*> node_map(map_->info.height * map_->info.width, nullptr);

        open_list.emplace(start.first, start.second, 0, heuristic(start, goal));
        node_map[start.second * map_->info.width + start.first] = new AStarNode(start.first, start.second, 0, 0);

        std::vector<MapPoint> path;
        bool found = false;

        while (!open_list.empty()) {
            AStarNode current = open_list.top();
            open_list.pop();

            if (closed_list[current.y][current.x]) continue;
            closed_list[current.y][current.x] = true;

            if (current.x == goal.first && current.y == goal.second) {
                path = reconstructPath(current);
                found = true;
                break;
            }

            // Generate neighbors
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;
                    if (!allow_diagonal_ && abs(dx) + abs(dy) > 1) continue;

                    int nx = current.x + dx;
                    int ny = current.y + dy;

                    if (!isValid(nx, ny)) continue;
                    if (closed_list[ny][nx]) continue;

                    double move_cost = (dx == 0 || dy == 0) ? 1.0 : 1.414;
                    double new_g = current.g + move_cost;
                    double new_h = heuristic({nx, ny}, goal);
                    AStarNode* neighbor = new AStarNode(nx, ny, new_g, new_h, node_map[current.y * map_->info.width + current.x]);

                    if (!node_map[ny * map_->info.width + nx] || new_g < node_map[ny * map_->info.width + nx]->g) {
                        open_list.push(*neighbor);
                        if (node_map[ny * map_->info.width + nx]) delete node_map[ny * map_->info.width + nx];
                        node_map[ny * map_->info.width + nx] = neighbor;
                    }
                }
            }
        }

        // Cleanup
        for (auto& n : node_map) if (n) delete n;
        
        return path;
    }

    std::vector<MapPoint> reconstructPath(const AStarNode& node) {
        std::vector<MapPoint> path;
        const AStarNode* current = &node;
        while (current != nullptr) {
            auto wp = gridToWorld(current->x, current->y);
            path.push_back({wp.first, wp.second});
            current = current->parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    double heuristic(std::pair<int, int> a, std::pair<int, int> b) {
        if (heuristic_type_ == "manhattan") {
            return abs(a.first - b.first) + abs(a.second - b.second);
        }
        return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
    }

    bool isValid(int x, int y) {
        if (x < 0 || y < 0 || x >= (int)map_->info.width || y >= (int)map_->info.height) {
            return false;
        }
        return map_->data[y * map_->info.width + x] == 0;
    }

    void publishPath(const std::vector<MapPoint>& path) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.1;
        marker.color.a = 1.0;
        marker.color.b = 1.0;

        for (const auto& p : path) {
            geometry_msgs::msg::Point point;
            point.x = p.x;
            point.y = p.y;
            marker.points.push_back(point);
        }

        path_pub_->publish(marker);
    }

    void publishSearchVisualization() {
      // TODO:(sel) implement the visualization of open and closed lists
    }

    void publishStatus() {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.z = 0.3; // Text size
        marker.color.a = 1.0; // Fully opaque
        marker.color.r = 1.0; // White color
        marker.color.g = 1.0;
        marker.color.b = 1.0;

        if (!current_pose_) {
            marker.text = "Status: Waiting for pose";
        } else if (!goal_point_) {
            marker.text = "Status: Waiting for goal";
        } else if (goal_reached_) {
            marker.text = "Status: Goal reached";
        } else if (current_path_.empty()) {
            marker.text = "Status: No path found";
        } else {
            marker.text = "Status: Following path";
        }

        marker.pose.position.x = current_pose_->x;
        marker.pose.position.y = current_pose_->y;
        marker.pose.position.z = 1.0; // Above the robot

        status_pub_->publish(marker);
    }

    void controlLoop() {
        if (goal_reached_ || current_path_.empty()) return;

        // Check goal proximity
        double dx = current_pose_->x - goal_point_->x;
        double dy = current_pose_->y - goal_point_->y;
        if (sqrt(dx*dx + dy*dy) < goal_tolerance_) {
            RCLCPP_INFO(get_logger(), "Goal reached! Stopping.");
            publishStopCommand();
            goal_reached_ = true;
            current_path_.clear();
            return;
        }

        // TODO(sel): add pure pursuit 
    }

    void publishStopCommand() {
        ackermann_msgs::msg::AckermannDriveStamped stop_msg;
        stop_msg.drive.speed = 0.0;
        drive_pub_->publish(stop_msg);
    }

    std::pair<int, int> worldToGrid(double x, double y) {
        int gx = (x - map_->info.origin.position.x) / map_->info.resolution;
        int gy = (y - map_->info.origin.position.y) / map_->info.resolution;
        return {gx, gy};
    }

    std::pair<double, double> gridToWorld(int x, int y) {
        double wx = x * map_->info.resolution + map_->info.origin.position.x;
        double wy = y * map_->info.resolution + map_->info.origin.position.y;
        return {wx, wy};
    }

    MapPoint toMapPoint(const geometry_msgs::msg::Pose& pose) {
        return {pose.position.x, pose.position.y};
    }

    // Member variables
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr open_list_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr closed_list_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr status_pub_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    nav_msgs::msg::OccupancyGrid::SharedPtr map_;
    std::optional<MapPoint> current_pose_;
    std::optional<MapPoint> goal_point_;
    std::vector<MapPoint> current_path_;
    bool goal_reached_ = false;

    // Parameters
    std::string heuristic_type_;
    bool allow_diagonal_;
    double lookahead_distance_;
    double speed_;
    double goal_tolerance_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr planning_timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AStarPlanner>());
    rclcpp::shutdown();
    return 0;
}