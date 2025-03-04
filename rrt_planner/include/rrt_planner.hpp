#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>
#include <cmath>
#include <cstdlib>

class RRTPlanner {
public:
    struct Node {
        geometry_msgs::msg::Point point;
        int parent_idx;
        double cost;

        Node(geometry_msgs::msg::Point p = {}, int parent = -1, double c = 0.0)
            : point(p), parent_idx(parent), cost(c) {}
    };

    struct PlanResult {
        std::vector<geometry_msgs::msg::Point> path;
        std::vector<Node> tree;
        std::vector<geometry_msgs::msg::Point> samples;
    };

    RRTPlanner(double step, double tolerance, int max_iter, double bias)
        : step_size(step), goal_tolerance(tolerance),
          max_iterations(max_iter), goal_bias(bias) {}

    PlanResult plan(const geometry_msgs::msg::Pose& start,
                   const geometry_msgs::msg::Pose& goal,
                   const nav_msgs::msg::OccupancyGrid& map);

private:
    double step_size;
    double goal_tolerance;
    int max_iterations;
    double goal_bias;

    geometry_msgs::msg::Point random_sample(const nav_msgs::msg::OccupancyGrid& map);
    int nearest_node(const geometry_msgs::msg::Point& sample, const std::vector<Node>& tree);
    geometry_msgs::msg::Point steer(const geometry_msgs::msg::Point& from, const geometry_msgs::msg::Point& to);
    bool check_collision(const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b, const nav_msgs::msg::OccupancyGrid& map);
    std::vector<std::pair<int, int>> bresenham_line(int x0, int y0, int x1, int y1);
    double distance(const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b);
    std::vector<geometry_msgs::msg::Point> trace_path(const Node& end_node, const std::vector<Node>& tree);
};

class RRTNode : public rclcpp::Node {
public:
    RRTNode();

private:
    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void plan();

    void visualize_tree(const std::vector<RRTPlanner::Node>& tree);
    void visualize_samples(const std::vector<geometry_msgs::msg::Point>& samples);
    void visualize_path(const std::vector<geometry_msgs::msg::Point>& path);
    void visualize_status(bool success);
    void timer_callback();

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tree_nodes_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tree_edges_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr samples_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr status_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    nav_msgs::msg::OccupancyGrid current_map_;
    geometry_msgs::msg::Pose current_pose_;
    geometry_msgs::msg::Pose goal_pose_;
    
    bool map_available_ = false;
    bool pose_available_ = false;
    bool goal_available_ = false;
    
    std::unique_ptr<RRTPlanner> planner_;
};