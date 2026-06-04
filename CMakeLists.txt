cmake_minimum_required(VERSION 3.8)
project(rl_balancer_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ddsm115_driver REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BalancerStatus.msg"
  DEPENDENCIES std_msgs
)

# ONNX Runtime — bundled local copy
set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)

set(NODE_DEPS
  rclcpp
  rclcpp_components
  sensor_msgs
  nav_msgs
  geometry_msgs
  ddsm115_driver
  std_msgs
  tf2_ros
  tf2_geometry_msgs
)

# Component library
add_library(rl_balancer_node_component SHARED src/rl_balancer_node.cpp)
target_compile_definitions(rl_balancer_node_component PRIVATE COMPONENT_ONLY)
target_include_directories(rl_balancer_node_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_ROOT}/include)
target_link_directories(rl_balancer_node_component PUBLIC ${ONNXRUNTIME_ROOT}/lib)
ament_target_dependencies(rl_balancer_node_component ${NODE_DEPS})
target_link_libraries(rl_balancer_node_component onnxruntime)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(rl_balancer_node_component "${cpp_typesupport_target}")
rclcpp_components_register_nodes(rl_balancer_node_component "RLBalancerNode")

# Standalone executable (target renamed to avoid clash with rosidl utility target)
add_executable(rl_balancer_node_exec src/rl_balancer_node.cpp)
set_target_properties(rl_balancer_node_exec PROPERTIES OUTPUT_NAME rl_balancer_node)
target_include_directories(rl_balancer_node_exec PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_ROOT}/include)
target_link_directories(rl_balancer_node_exec PUBLIC ${ONNXRUNTIME_ROOT}/lib)
ament_target_dependencies(rl_balancer_node_exec ${NODE_DEPS})
target_link_libraries(rl_balancer_node_exec onnxruntime "${cpp_typesupport_target}")

install(TARGETS rl_balancer_node_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS rl_balancer_node_exec
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY policy/
  DESTINATION share/${PROJECT_NAME}/policy)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

# Install onnxruntime shared libs alongside the node
install(DIRECTORY ${ONNXRUNTIME_ROOT}/lib/
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
