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
#include <unordered_map>
#include <vector>

using namespace std::chrono_literals;

struct HASNode {
    double x, y, theta;
    double g, h, f;
    HASNode* parent;
    double steering;

    HASNode(double x, double y, double theta, double g, double h, HASNode* parent = nullptr, double steering = 0)
        : x(x), y(y), theta(theta), g(g), h(h), f(g + h), parent(parent), steering(steering) {}

    bool operator<(const HASNode& other) const {
        return f > other.f; // For priority queue (min-heap)
    }
};

struct Pose {
    double x, y, theta;
};

class HybridAStarPlanner : public rclcpp::Node {
public:
    HybridAStarPlanner() : Node("hybrid_a_star_planner") {
        // Parameters
        declare_parameter("max_steering_angle", 0.34);
        declare_parameter("step_length", 0.1);
        declare_parameter("wheelbase", 0.3302);
        declare_parameter("width", 0.2032);
        declare_parameter("lookahead_distance", 1.0);
        declare_parameter("speed", 2.0);

        max_steering_angle_ = get_parameter("max_steering_angle").as_double();
        step_length_ = get_parameter("step_length").as_double();
        wheelbase_ = get_parameter("wheelbase").as_double();
        width_ = get_parameter("width").as_double();
        lookahead_distance_ = get_parameter("lookahead_distance").as_double();
        speed_ = get_parameter("speed").as_double();

        // Subscribers
        map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10, std::bind(&HybridAStarPlanner::mapCallback, this, std::placeholders::_1));
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10, std::bind(&HybridAStarPlanner::odomCallback, this, std::placeholders::_1));
        goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10, std::bind(&HybridAStarPlanner::goalCallback, this, std::placeholders::_1));

        // Publishers
        drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        path_pub_ = create_publisher<visualization_msgs::msg::Marker>("/path", 10);
        expanded_pub_ = create_publisher<visualization_msgs::msg::Marker>("/expanded_nodes", 10);

        // TF
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Timer for control loop
        timer_ = create_wall_timer(50ms, std::bind(&HybridAStarPlanner::controlLoop, this));
    }

private:
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        map_ = msg;
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        geometry_msgs::msg::PoseStamped odom_pose;
        odom_pose.header = msg->header;
        odom_pose.pose = msg->pose.pose;

        try {
            auto map_pose = tf_buffer_->transform(odom_pose, "map", tf2::durationFromSec(0.1));
            current_pose_ = toPose(map_pose.pose);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(get_logger(), "Transform error: %s", ex.what());
        }
    }

    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        goal_pose_ = toPose(msg->pose);
        planPath();
    }

    void planPath() {
        if (!map_ || !current_pose_ || !goal_pose_) return;

        auto path = hybridAStar(*current_pose_, *goal_pose_);
        publishPath(path);
        current_path_ = path;
    }

    std::vector<Pose> hybridAStar(const Pose& start, const Pose& goal) {
        std::priority_queue<HASNode> open_list;
        std::unordered_map<std::string, double> closed_list;

        open_list.emplace(start.x, start.y, start.theta, 0.0, heuristic(start, goal));

        std::vector<Pose> path;

        while (!open_list.empty()) {
            HASNode current = open_list.top();
            open_list.pop();

            std::string key = getKey(current.x, current.y, current.theta);
            if (closed_list.find(key) != closed_list.end() && closed_list[key] <= current.f) {
                continue;
            }
            closed_list[key] = current.f;

            if (reachedGoal(current, goal)) {
                path = reconstructPath(current);
                break;
            }

            for (double steer : {-max_steering_angle_, 0.0, max_steering_angle_}) {
                for (double dir : {1.0, -1.0}) { // Forward and reverse
                    Pose next = move(current, steer, dir);
                    if (isCollision(next)) continue;

                    double g = current.g + step_length_;
                    double h = heuristic(next, goal);
                    HASNode next_node(next.x, next.y, next.theta, g, h, new HASNode(current), steer);

                    std::string next_key = getKey(next.x, next.y, next.theta);
                    if (closed_list.find(next_key) == closed_list.end() || g < closed_list[next_key]) {
                        open_list.push(next_node);
                    }
                }
            }
        }

        return path;
    }

    Pose move(const HASNode& node, double steer, double dir) {
        double delta_theta = (dir * step_length_ / wheelbase_) * std::tan(steer);
        double theta = node.theta + delta_theta;
        double x = node.x + dir * step_length_ * std::cos(node.theta);
        double y = node.y + dir * step_length_ * std::sin(node.theta);
        return {x, y, theta};
    }

    bool isCollision(const Pose& pose) {
        // Check footprint
        std::vector<std::pair<double, double>> footprint = getFootprint(pose);
        for (const auto& point : footprint) {
            int mx = static_cast<int>((point.first - map_->info.origin.position.x) / map_->info.resolution);
            int my = static_cast<int>((point.second - map_->info.origin.position.y) / map_->info.resolution);
            if (mx < 0 || my < 0 || mx >= (int)map_->info.width || my >= (int)map_->info.height) {
                return true;
            }
            int index = my * map_->info.width + mx;
            if (map_->data[index] > 50 || map_->data[index] == -1) {
                return true;
            }
        }
        return false;
    }

    std::vector<std::pair<double, double>> getFootprint(const Pose& pose) {
        std::vector<std::pair<double, double>> footprint;
        double half_width = width_ / 2.0;
        double length = wheelbase_;

        // Four corners relative to base_link
        std::vector<std::pair<double, double>> relative_points = {
            {0, -half_width}, {0, half_width},
            {length, -half_width}, {length, half_width}
        };

        for (const auto& p : relative_points) {
            double x = pose.x + p.first * std::cos(pose.theta) - p.second * std::sin(pose.theta);
            double y = pose.y + p.first * std::sin(pose.theta) + p.second * std::cos(pose.theta);
            footprint.emplace_back(x, y);
        }

        return footprint;
    }

    double heuristic(const Pose& a, const Pose& b) {
        return std::hypot(a.x - b.x, a.y - b.y);
    }

    bool reachedGoal(const HASNode& node, const Pose& goal) {
        return std::hypot(node.x - goal.x, node.y - goal.y) < 0.1;
    }

    std::vector<Pose> reconstructPath(const HASNode& node) {
        std::vector<Pose> path;
        const HASNode* current = &node;
        while (current != nullptr) {
            path.push_back({current->x, current->y, current->theta});
            current = current->parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    void publishPath(const std::vector<Pose>& path) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.1;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;

        for (const auto& pose : path) {
            geometry_msgs::msg::Point p;
            p.x = pose.x;
            p.y = pose.y;
            p.z = 0.0;
            marker.points.push_back(p);
        }

        path_pub_->publish(marker);
    }

    void controlLoop() {
        if (current_path_.empty()) return;

        // Pure Pursuit
        Pose current = *current_pose_;
        double min_dist = std::numeric_limits<double>::max();
        int target_index = 0;

        // Find closest point on path
        for (size_t i = 0; i < current_path_.size(); ++i) {
            double dist = std::hypot(current_path_[i].x - current.x, current_path_[i].y - current.y);
            if (dist < min_dist) {
                min_dist = dist;
                target_index = i;
            }
        }

        // Find lookahead point
        double lookahead = lookahead_distance_;
        for (size_t i = target_index; i < current_path_.size(); ++i) {
            double dist = std::hypot(current_path_[i].x - current.x, current_path_[i].y - current.y);
            if (dist >= lookahead) {
                target_index = i;
                break;
            }
        }

        // Calculate steering angle
        Pose target = current_path_[target_index];
        double alpha = std::atan2(target.y - current.y, target.x - current.x) - current.theta;
        double steering = std::atan2(2.0 * wheelbase_ * std::sin(alpha), lookahead_distance_);

        // Publish drive command
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.drive.steering_angle = steering;
        drive_msg.drive.speed = speed_;
        drive_pub_->publish(drive_msg);
    }

    std::string getKey(double x, double y, double theta) {
        const double precision = 10.0;
        return std::to_string(static_cast<int>(x * precision)) + "_" +
               std::to_string(static_cast<int>(y * precision)) + "_" +
               std::to_string(static_cast<int>(theta * precision));
    }

    Pose toPose(const geometry_msgs::msg::Pose& pose) {
        tf2::Quaternion q(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        return {pose.position.x, pose.position.y, yaw};
    }

    // Members
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr expanded_pub_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    nav_msgs::msg::OccupancyGrid::SharedPtr map_;
    std::optional<Pose> current_pose_;
    std::optional<Pose> goal_pose_;
    std::vector<Pose> current_path_;

    double max_steering_angle_;
    double step_length_;
    double wheelbase_;
    double width_;
    double lookahead_distance_;
    double speed_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HybridAStarPlanner>());
    rclcpp::shutdown();
    return 0;
}
