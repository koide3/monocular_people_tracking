cmake_minimum_required(VERSION 2.8.3)
project(monocular_people_tracking)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  cv_bridge
  tf_conversions
  image_transport
  tfpose_ros
)

########################
## message generation ##
########################
add_message_files(FILES
  Track.msg
  TrackArray.msg
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs tfpose_ros)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES monocular_people_tracking
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


###########
## Build ##
###########
include_directories(
  include
  3rdparty/CppNumericalSolvers/include
  ${catkin_INCLUDE_DIRS}
)

add_executable(monocular_people_tracking_node
  src/track_system.cpp
  src/person_tracker.cpp
  src/people_tracker.cpp
  src/monocular_people_tracking_node.cpp
)
target_link_libraries(monocular_people_tracking_node
  ${catkin_LIBRARIES}
)
add_dependencies(monocular_people_tracking_node monocular_people_tracking_gencpp)
