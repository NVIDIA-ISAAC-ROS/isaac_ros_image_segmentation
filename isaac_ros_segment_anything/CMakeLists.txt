# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_segment_anything LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
enable_language(CUDA)
# Decoder node

ament_auto_add_library(segment_anything_decoder_node SHARED src/segment_anything_decoder_node.cpp)
rclcpp_components_register_nodes(segment_anything_decoder_node "nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode")
set(node_plugins
"${node_plugins}nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode;$<TARGET_FILE:segment_antyhing_decoder_node>\n")
set_target_properties(segment_anything_decoder_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

ament_auto_add_library(segment_anything_data_encoder_node SHARED src/segment_anything_data_encoder_node.cpp)
rclcpp_components_register_nodes(segment_anything_data_encoder_node "nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode")
set(node_plugins
"${node_plugins}nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode;$<TARGET_FILE:segment_anything_data_encoder_node>\n")
set_target_properties(segment_anything_data_encoder_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

ament_auto_add_library(segment_anything_dummy_mask_publisher_node SHARED src/segment_anything_dummy_mask_publisher_node.cpp)
rclcpp_components_register_nodes(segment_anything_dummy_mask_publisher_node "nvidia::isaac_ros::segment_anything::DummyMaskPublisher")
set(node_plugins
"${node_plugins}nvidia::isaac_ros::segment_anything::DummyMaskPublisher;$<TARGET_FILE:segment_anything_dummy_mask_publisher_node>\n")
set_target_properties(segment_anything_dummy_mask_publisher_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

ament_auto_add_library(segment_anything_point_triggered_node SHARED src/segment_anything_point_triggered_node.cpp)
rclcpp_components_register_nodes(
  segment_anything_point_triggered_node "nvidia::isaac_ros::segment_anything::SegmentAnythingPointTriggeredNode")
set(node_plugins
"${node_plugins}nvidia::isaac_ros::segment_anything::SegmentAnythingPointTriggeredNode;$<TARGET_FILE:segment_anything_point_triggered_node>\n")
set_target_properties(segment_anything_point_triggered_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Segment Anything Tensor to Image
ament_auto_add_library(segment_anything_tensor_to_image_node SHARED
  src/segment_anything_tensor_to_image_node.cpp
  src/segment_anything_binarize_tensor.cu)
  rclcpp_components_register_nodes(segment_anything_tensor_to_image_node
    "nvidia::isaac_ros::segment_anything::TensorToImageNode")
  set(node_plugins
    "${node_plugins}nvidia::isaac_ros::segment_anything::TensorToImageNode;$<TARGET_FILE:segment_anything_tensor_to_image_node>\n")
  set_target_properties(segment_anything_tensor_to_image_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  # Mark as CUDA files with non-standard extensions
  set_source_files_properties(
    src/segment_anything_binarize_tensor.cu
    include/isaac_ros_segment_anything/segment_anything_binarize_tensor.hpp
    PROPERTIES LANGUAGE CUDA
  )

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/colored_mask_converter_node.py DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/torch_to_onnx.py DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/visualize_mask.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Force use of ROS2 vendor googletest to avoid version conflicts
  set(CMAKE_IGNORE_PATH "/usr/src/googletest" ${CMAKE_IGNORE_PATH})
  set(CMAKE_IGNORE_PATH "/usr/include/gtest" ${CMAKE_IGNORE_PATH})
  set(CMAKE_IGNORE_PATH "/usr/include/gmock" ${CMAKE_IGNORE_PATH})

  # Explicitly use ROS2 vendor googletest/gmock
  include_directories(BEFORE /opt/ros/$ENV{ROS_DISTRO}/src/gtest_vendor/include)
  include_directories(BEFORE /opt/ros/$ENV{ROS_DISTRO}/src/gmock_vendor/include)

  # Gtests for segment anything data encoder node
  ament_add_gtest(segment_anything_data_encoder_node_test test/segment_anything_data_encoder_node_test.cpp)
  target_link_libraries(segment_anything_data_encoder_node_test segment_anything_data_encoder_node)
  target_include_directories(segment_anything_data_encoder_node_test PUBLIC include/isaac_ros_segment_anything/)
  ament_target_dependencies(segment_anything_data_encoder_node_test rclcpp)
  ament_target_dependencies(segment_anything_data_encoder_node_test isaac_ros_nitros)

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_segment_anything_onnx_pol.py TIMEOUT "300")
  add_launch_test(test/isaac_ros_segment_anything_point_prompt_pol.py TIMEOUT "300")
  add_launch_test(test/isaac_ros_segment_anything_tensor_to_image_test.py)
endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE config launch)
