cmake_minimum_required(VERSION 2.4.6)
project(dso_ros)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  nav_msgs
  cv_bridge
  tf2_ros
  image_transport
  fla_msgs
)

if (DEFINED ENV{DSO_PATH})
  set(DSO_PATH $ENV{DSO_PATH})
elseif (DEFINED ENV{GTSC_PREFIX_PATH})
  set(DSO_PATH $ENV{GTSC_PREFIX_PATH}/../dso)
else ()
  message(FATAL_WARNING "Could not find DSO!")
endif ()
message("\n\n---- LOOKING FOR DSO at\n\"${DSO_PATH}\"")


find_package(Pangolin 0.2 REQUIRED)
message("\n\n---- FOUND Pangolin Headers at\n\"${Pangolin_INCLUDE_DIRS}\"")
message("---- FOUND Pangolin Libs at\n\"${Pangolin_LIBRARIES}\"")

find_package(OpenCV REQUIRED)
message("\n\n---- FOUND OpenCV Headers at\n\"${OpenCV_INCLUDE_DIRS}\"")
message("---- FOUND OpenCV Libs at\n\"${OpenCV_LIBS}\"")

# find_package(Eigen3 REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 REQUIRED eigen3>=3.2)

find_package(Boost COMPONENTS system thread)
find_library(DSO_LIBRARY dso ${DSO_PATH}/build/lib)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    roscpp
    sensor_msgs
    nav_msgs
    cv_bridge
    tf2_ros
    image_transport
    pcl_ros
    pcl_msgs
    pcl_conversions
    fla_msgs)

find_package(PCL 1.7 REQUIRED COMPONENTS common)

###########
## Build ##
###########

set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O2 -g -std=c++0x -march=native -fno-omit-frame-pointer"
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${DSO_PATH}/src
  ${DSO_PATH}/thirdparty/Sophus
  ${Pangolin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(dso_live
  src/main.cpp
  src/ROSOutputWrapper.h
  src/utils.h)

target_compile_definitions(dso_live PUBLIC ${PCL_DEFINITIONS})

add_dependencies(dso_live fla_msgs_generate_messages_cpp)

target_link_libraries(dso_live
    ${DSO_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES})
