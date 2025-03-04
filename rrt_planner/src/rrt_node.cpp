#include "rrt_planner.hpp"
#include <chrono>

using namespace std::chrono_literals;

RRTPlanner::PlanResult RRTPlanner::plan(const geometry_msgs::msg::Pose& start,
                                       const geometry_msgs::msg::Pose& goal,
                                       const nav_msgs::msg::OccupancyGrid& map) {
    PlanResult result;
    std::vector<Node> tree;
    std::vector<geometry_msgs::msg::Point> samples;

    tree.emplace_back(start.position);

    for (int i = 0; i < max_iterations; ++i) {
        geometry_msgs::msg::Point sample;
        if ((rand() % 100) < (goal_bias * 100)) {
            sample = goal.position;
        } else {
            sample = random_sample(map);
        }
        samples.push_back(sample);

        int nearest_idx = nearest_node(sample, tree);
        geometry_msgs::msg::Point new_point = steer(tree[nearest_idx].point, sample);

        if (!check_collision(tree[nearest_idx].point, new_point, map)) {
            Node new_node;
            new_node.point = new_point;
            new_node.parent_idx = nearest_idx;
            new_node.cost = tree[nearest_idx].cost + distance(tree[nearest_idx].point, new_point);
            tree.push_back(new_node);

            if (distance(new_point, goal.position) <= goal_tolerance) {
                result.path = trace_path(new_node, tree);
                break;
            }
        }
    }

    result.tree = tree;
    result.samples = samples;
    return result;
}

geometry_msgs::msg::Point RRTPlanner::random_sample(const nav_msgs::msg::OccupancyGrid& map) {
    geometry_msgs::msg::Point p;
    double x0 = map.info.origin.position.x;
    double y0 = map.info.origin.position.y;
    double x_range = map.info.width * map.info.resolution;
    double y_range = map.info.height * map.info.resolution;

    p.x = x0 + (rand() / (double)RAND_MAX) * x_range;
    p.y = y0 + (rand() / (double)RAND_MAX) * y_range;
    return p;
}

int RRTPlanner::nearest_node(const geometry_msgs::msg::Point& sample, const std::vector<Node>& tree) {
    int nearest = 0;
    double min_dist = INFINITY;
    for (size_t i = 0; i < tree.size(); ++i) {
        double d = distance(tree[i].point, sample);
        if (d < min_dist) {
            min_dist = d;
            nearest = i;
        }
    }
    return nearest;
}

geometry_msgs::msg::Point RRTPlanner::steer(const geometry_msgs::msg::Point& from, const geometry_msgs::msg::Point& to) {
    double dx = to.x - from.x;
    double dy = to.y - from.y;
    double dist = hypot(dx, dy);
    
    if (dist <= step_size) return to;
    
    geometry_msgs::msg::Point p;
    p.x = from.x + (dx / dist) * step_size;
    p.y = from.y + (dy / dist) * step_size;
    return p;
}

bool RRTPlanner::check_collision(const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b, const nav_msgs::msg::OccupancyGrid& map) {
    int x0 = static_cast<int>((a.x - map.info.origin.position.x) / map.info.resolution);
    int y0 = static_cast<int>((a.y - map.info.origin.position.y) / map.info.resolution);
    int x1 = static_cast<int>((b.x - map.info.origin.position.x) / map.info.resolution);
    int y1 = static_cast<int>((b.y - map.info.origin.position.y) / map.info.resolution);

    auto line = bresenham_line(x0, y0, x1, y1);
    for (const auto& cell : line) {
        if (cell.first < 0 || cell.first >= map.info.width || cell.second < 0 || cell.second >= map.info.height)
            return true;
        
        int idx = cell.second * map.info.width + cell.first;
        if (map.data[idx] > 50) return true;
    }
    return false;
}

std::vector<std::pair<int, int>> RRTPlanner::bresenham_line(int x0, int y0, int x1, int y1) {
    std::vector<std::pair<int, int>> points;
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (true) {
        points.emplace_back(x0, y0);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
    return points;
}

double RRTPlanner::distance(const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b) {
    return hypot(a.x - b.x, a.y - b.y);
}

std::vector<geometry_msgs::msg::Point> RRTPlanner::trace_path(const Node& end_node, const std::vector<Node>& tree) {
    std::vector<geometry_msgs::msg::Point> path;
    const Node* current = &end_node;
    while (current->parent_idx != -1) {
        path.push_back(current->point);
        current = &tree[current->parent_idx];
    }
    path.push_back(current->point);
    std::reverse(path.begin(), path.end());
    return path;
}

RRTNode::RRTNode() : Node("rrt_planner") {
    auto map_qos = rclcpp::QoS(rclcpp::KeepLast(10)).transient_local().reliable();
    
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", map_qos, std::bind(&RRTNode::map_callback, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/ego_racecar/odom", 10, std::bind(&RRTNode::odom_callback, this, std::placeholders::_1));
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 10, std::bind(&RRTNode::goal_callback, this, std::placeholders::_1));

    tree_nodes_pub_ = create_publisher<visualization_msgs::msg::Marker>("/rrt/nodes", 10);
    tree_edges_pub_ = create_publisher<visualization_msgs::msg::Marker>("/rrt/edges", 10);
    samples_pub_ = create_publisher<visualization_msgs::msg::Marker>("/rrt/samples", 10);
    path_pub_ = create_publisher<visualization_msgs::msg::Marker>("/rrt/path", 10);
    status_pub_ = create_publisher<visualization_msgs::msg::Marker>("/rrt/status", 10);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    planner_ = std::make_unique<RRTPlanner>(0.1, 0.15, 5000, 0.65);
    

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50ms),
      std::bind(&RRTNode::timer_callback, this)
    );

}

void RRTNode::timer_callback() {
  RCLCPP_INFO(get_logger(), "Timer callback triggered!");
  RCLCPP_INFO(get_logger(), "Pose Available: %d, Goal Available: %d, Map Available: %d", pose_available_, goal_available_, map_available_);
  if (map_available_ && pose_available_ && goal_available_) {
      plan();
  }
}

void RRTNode::map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "Received a new map");
    current_map_ = *msg;
    map_available_ = true;
}

void RRTNode::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    geometry_msgs::msg::PoseStamped odom_pose;
    odom_pose.header = msg->header;
    odom_pose.pose = msg->pose.pose;

    try {
        geometry_msgs::msg::PoseStamped map_pose = tf_buffer_->transform(
            odom_pose, "map", tf2::durationFromSec(0.1));
        current_pose_ = map_pose.pose;
        pose_available_ = true;
        // RCLCPP_INFO(get_logger(), "Pose available: %d", pose_available_);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(get_logger(), "Transform error: %s", ex.what());
    }
}

void RRTNode::goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "Received a new goal");
    goal_pose_ = msg->pose;
    goal_available_ = true;
}

void RRTNode::plan() {
    RCLCPP_INFO(get_logger(), "Planning...");
    auto result = planner_->plan(current_pose_, goal_pose_, current_map_);
    RCLCPP_INFO(get_logger(), "Planning finished. Result: %s", result.path.empty() ? "Failed" : "Success");
    
    visualize_tree(result.tree);
    visualize_samples(result.samples);
    visualize_path(result.path);
    visualize_status(!result.path.empty());
}

void RRTNode::visualize_tree(const std::vector<RRTPlanner::Node>& tree) {
    visualization_msgs::msg::Marker nodes, edges;
    nodes.header = edges.header = current_map_.header;
    nodes.ns = edges.ns = "rrt";
    nodes.id = 0; edges.id = 1;
    
    nodes.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.1;
    nodes.color.g = 1.0; nodes.color.a = 1.0;
    
    edges.type = visualization_msgs::msg::Marker::LINE_LIST;
    edges.scale.x = 0.05;
    edges.color.g = 0.5; edges.color.a = 0.8;

    for (size_t i = 0; i < tree.size(); ++i) {
        nodes.points.push_back(tree[i].point);
        if (tree[i].parent_idx >= 0) {
            edges.points.push_back(tree[tree[i].parent_idx].point);
            edges.points.push_back(tree[i].point);
        }
    }

    tree_nodes_pub_->publish(nodes);
    tree_edges_pub_->publish(edges);
}

void RRTNode::visualize_samples(const std::vector<geometry_msgs::msg::Point>& samples) {
    visualization_msgs::msg::Marker marker;
    marker.header = current_map_.header;
    marker.ns = "samples";
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.scale.x = marker.scale.y = marker.scale.z = 0.1;
    marker.color.r = 1.0; marker.color.a = 0.5;
    
    marker.points = samples;
    samples_pub_->publish(marker);
}

void RRTNode::visualize_path(const std::vector<geometry_msgs::msg::Point>& path) {
    visualization_msgs::msg::Marker marker;
    marker.header = current_map_.header;
    marker.ns = "path";
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.scale.x = 0.1;
    marker.color.b = 1.0; marker.color.a = 1.0;
    
    marker.points = path;
    path_pub_->publish(marker);
}

void RRTNode::visualize_status(bool success) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.scale.z = 0.5;
    marker.color.r = marker.color.g = marker.color.b = 1.0;
    marker.color.a = 1.0;
    marker.pose.position = current_pose_.position;
    marker.pose.position.z += 1.0;
    marker.text = success ? "Path Found!" : "Planning Failed";
    
    status_pub_->publish(marker);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RRTNode>());
    rclcpp::shutdown();
    return 0;
}