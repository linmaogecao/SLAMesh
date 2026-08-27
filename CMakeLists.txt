cmake_minimum_required(VERSION 2.8.3)
project(slamesh)

SET(CMAKE_CXX_FLAGS "-std=c++17 -Wno-deprecated-declarations")
set(CMAKE_BUILD_TYPE "Release")
set(catkin_DIR /opt/ros/kinetic/share/catkin/cmake)

#ros
#add_compile_definitions(VOXBLOX)
#set(USE_VOXBLOX ON)
set(USE_VOXBLOX OFF)
if(USE_VOXBLOX)
    message("USE_VOXBLOX")
    find_package(catkin REQUIRED COMPONENTS
            pcl_ros
            roscpp
            rospy
            sensor_msgs
            std_msgs
            mesh_msgs
            mesh_msgs_conversions
            )
else()
    message("DONT USE_VOXBLOX")
    find_package(catkin REQUIRED COMPONENTS
            pcl_ros
            roscpp
            rospy
            sensor_msgs
            std_msgs
            mesh_msgs
            mesh_msgs_conversions
            )
endif()
catkin_package()
include_directories(include ${catkin_INCLUDE_DIRS})

#pcl
find_package(PCL REQUIRED)
include_directories(include ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
#ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_path(ANN_INCLUDE_DIR NAMES ANN/ANN.h PATHS /usr/include /usr/local/include)
find_library(ANN_LIBRARY NAMES ANN ann PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

if(NOT ANN_INCLUDE_DIR OR NOT ANN_LIBRARY)
    message(FATAL_ERROR "ANN library not found! Install libann-dev or set ANN_INCLUDE_DIR and ANN_LIBRARY manually.")
else()
    message(STATUS "Found ANN: include=${ANN_INCLUDE_DIR}, lib=${ANN_LIBRARY}")
endif()
include_directories(${ANN_INCLUDE_DIR})

#opencv in vins
#find_package(OpenCV REQUIRED)
#ictmap
#find_package(octomap REQUIRED)
#include_directories(${OCTOMAP_INCLUDE_DIRS})
#threads (建图线程 / shared_mutex)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
#mesh_msg
find_package(LVR2 25 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
include_directories(include ${LVR2_INCLUDE_DIRS})
if(USE_VOXBLOX)
    message("USE_VOXBLOX")
    add_executable(slamesh
            src/utility/tool.cpp
            src/factor/lidarFactor.cpp
            src/cell.cpp
            src/map.cpp
            src/slamesher_node.cpp
            src/voxblox/mesh/mesher.cpp
            src/voxblox/mesh/marching_cubes.cc
            src/RangeMap.cpp
            src/BSplineSurface.cpp
            src/readWrite.cpp
            src/tool.cpp
            )
else()
    message("DONT USE_VOXBLOX")
    add_executable(slamesh
            src/utility/tool.cpp
            src/factor/lidarFactor.cpp
            src/cell.cpp
            src/map.cpp
            src/slamesher_node.cpp
            src/RangeMap.cpp
            src/BSplineSurface.cpp
            src/readWrite.cpp
            src/tool.cpp
            )
endif()

#target_link_libraries(slamesh ${catkin_LIBRARIES} ${PCL_LIBRARIES}  ${CERES_LIBRARIES} gflags ${GFLAGS_LIBRARY}
#        ${LVR2_LIBRARIES})
target_link_libraries(slamesh ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS} ${CERES_LIBRARIES} ${ANN_LIBRARY} ${LVR2_LIBRARIES} Threads::Threads )





