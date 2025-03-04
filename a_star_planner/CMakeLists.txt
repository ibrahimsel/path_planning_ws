cmake_minimum_required(VERSION 3.8)
project(a_star_planner)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(a_star_node
  src/a_star_planner_node.cpp
)

ament_target_dependencies(a_star_node
  rclcpp
  nav_msgs
  geometry_msgs
  ackermann_msgs
  visualization_msgs
  tf2_ros
  tf2_geometry_msgs
)

install(TARGETS a_star_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()