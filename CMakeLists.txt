cmake_minimum_required(VERSION 2.8.3)
project(iplanner_node)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
# set(CMAKE_CXX_STANDARD 11)
# list(APPEND CMAKE_PREFIX_PATH "/usr/lib/libtorch")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  image_transport
  sensor_msgs
  geometry_msgs
  message_generation
)


# find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)



## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Msg1.msg
#   Msg2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs
#   std_msgs
# )

catkin_package(
#  INCLUDE_DIRS include
 CATKIN_DEPENDS 
 roscpp 
 rospy 
 std_msgs
 geometry_msgs
#  DEPENDS system_lib
)

catkin_install_python(PROGRAMS src/data_collect_node.py 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS src/iplanner_node.py 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS src/iplanner_viz.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
