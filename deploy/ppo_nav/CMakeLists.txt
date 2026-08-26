cmake_minimum_required(VERSION 3.0.2)
project(ppo_nav)

# gnu++14 以启用 M_PI 等 GNU 扩展（-std=c++14 会隐藏 M_PI 导致编译失败）
add_compile_options(-std=gnu++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
)

# planner_node 用 std::thread（事件驱动 worker），显式链接 pthread
find_package(Threads REQUIRED)

# ---- CUDA（Jetson: /usr/local/cuda）----
find_package(CUDA QUIET)
if(NOT CUDA_FOUND AND EXISTS "/usr/local/cuda")
  set(CUDA_FOUND TRUE)
  set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
  if(EXISTS "/usr/local/cuda/lib64/libcudart.so")
    set(CUDA_LIBRARIES /usr/local/cuda/lib64/libcudart.so)
  endif()
endif()
if(CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(EXTRA_LIBS ${CUDA_LIBRARIES})
else()
  message(WARNING "CUDA not found, cuda_runtime_api.h 可能缺失导致编译失败")
endif()

# ---- TensorRT (JetPack 自带: /usr/lib/aarch64-linux-gnu, /usr/include/aarch64-linux-gnu) ----
find_library(TRT_LIB NAMES nvinfer)
if(NOT TRT_LIB)
  find_path(_TRT_LIBDIR NAMES libnvinfer.so.8
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib NO_DEFAULT_PATH)
  if(_TRT_LIBDIR)
    set(TRT_LIB ${_TRT_LIBDIR}/libnvinfer.so.8)
  endif()
endif()
if(NOT TRT_LIB)
  message(FATAL_ERROR "libnvinfer not found. Install TensorRT (JetPack) or set TENSORRT_ROOT.")
endif()
find_path(TRT_INCLUDE_DIR NvInfer.h)
if(TRT_INCLUDE_DIR)
  include_directories(${TRT_INCLUDE_DIR})
else()
  include_directories(/usr/include/aarch64-linux-gnu)
endif()
message(STATUS "TensorRT: ${TRT_LIB}")

include_directories(include ${catkin_INCLUDE_DIRS})

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ppo_nav_lib
  CATKIN_DEPENDS roscpp sensor_msgs nav_msgs geometry_msgs std_msgs tf2_ros tf2_geometry_msgs
)

add_library(ppo_nav_lib
  src/tensorrt_engine.cpp
  src/obs_builder.cpp
)
target_link_libraries(ppo_nav_lib ${catkin_LIBRARIES} ${TRT_LIB} ${EXTRA_LIBS})

add_executable(planner_node src/planner_node.cpp)
target_link_libraries(planner_node ppo_nav_lib ${catkin_LIBRARIES} ${TRT_LIB} ${EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})

add_executable(safety_node src/safety_node.cpp)
target_link_libraries(safety_node ${catkin_LIBRARIES})

install(TARGETS planner_node safety_node ppo_nav_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
