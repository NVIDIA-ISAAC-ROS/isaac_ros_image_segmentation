# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(gxf_isaac_ros_segment_anything LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

# Dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(CUDAToolkit)
find_package(yaml-cpp)
find_package(isaac_ros_nitros_detection2_d_array_type REQUIRED)
enable_language(CUDA)
# Create extension
ament_auto_add_library(${PROJECT_NAME} SHARED
    gxf/segment_anything/segment_anything_ext.cpp
    gxf/segment_anything/segment_anything_postprocessor.cpp
    gxf/segment_anything/segment_anything_postprocessor.hpp
    gxf/segment_anything/segment_anything_postprocessor.cu.cpp
    gxf/segment_anything/segment_anything_postprocessor.cu.hpp
    gxf/segment_anything/segment_anything_prompt_processor.cpp
    gxf/segment_anything/segment_anything_prompt_processor.hpp
    gxf/segment_anything/segment_anything_msg_compositor.cpp
    gxf/segment_anything/segment_anything_msg_compositor.hpp
)

# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  gxf/segment_anything/segment_anything_postprocessor.cu.cpp
  gxf/segment_anything/segment_anything_postprocessor.cu.hpp
  PROPERTIES LANGUAGE CUDA
)
target_link_libraries(${PROJECT_NAME}
  CUDA::cudart
  yaml-cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/gxf"
${isaac_ros_nitros_detection2_d_array_type_INCLUDE_DIRS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install the binary file
install(TARGETS ${PROJECT_NAME} DESTINATION share/${PROJECT_NAME}/gxf/lib)

ament_auto_package(INSTALL_TO_SHARE)