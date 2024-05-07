cmake_minimum_required(VERSION 3.5)
project(dis_tutorial3)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# This is needed if we generate custom interfaces (messages, services or actions)
find_package(rosidl_default_generators REQUIRED)

# A macro, not to repeat the dependencies
set(dependencies rclcpp std_msgs std_srvs rosidl_default_generators)

# We list the interface files and their dependencies
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Coordinates.msg"
  DEPENDENCIES std_msgs # Add packages that the listed interfaces depend on
)

install(DIRECTORY
  launch
  config
  worlds
  maps
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/detect_people.py
  scripts/robot_commander.py
  scripts/map_goals.py
  scripts/transform_point.py
  scripts/detect_rings.py
  scripts/arm_mover_actions.py
  DESTINATION lib/${PROJECT_NAME})

# This is needed if our exported targets depend on interfaces built in the package
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

ament_export_dependencies(${dependencies} rosidl_default_runtime)

ament_package()
