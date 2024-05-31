# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(gxf_isaac_ros_unet LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Dependencies
find_package(CUDAToolkit)
find_package(yaml-cpp)

enable_language(CUDA)

# Create extension
ament_auto_add_library(${PROJECT_NAME} SHARED
  gxf/image_segmentation/segmentation_mask_colorizer.cpp
  gxf/image_segmentation/segmentation_mask_colorizer.hpp
  gxf/image_segmentation/segmentation_postprocessing_utils.hpp
  gxf/image_segmentation/segmentation_postprocessor_ext.cpp
  gxf/image_segmentation/segmentation_postprocessor.cpp
  gxf/image_segmentation/segmentation_postprocessor.hpp
  gxf/image_segmentation/segmentation_postprocessor.cu.cpp
  gxf/image_segmentation/segmentation_postprocessor.cu.hpp
  gxf/image_segmentation/segmentation_mask_colorizer.cu.cpp
  gxf/image_segmentation/segmentation_mask_colorizer.cu.hpp
)

# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  gxf/image_segmentation/segmentation_postprocessor.cu.cpp
  gxf/image_segmentation/segmentation_postprocessor.cu.hpp
  gxf/image_segmentation/segmentation_mask_colorizer.cu.cpp
  gxf/image_segmentation/segmentation_mask_colorizer.cu.hpp
  PROPERTIES LANGUAGE CUDA
)

target_link_libraries(${PROJECT_NAME}
  CUDA::cudart
  yaml-cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/gxf")

set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install the binary file
install(TARGETS ${PROJECT_NAME} DESTINATION share/${PROJECT_NAME}/gxf/lib)

ament_auto_package(INSTALL_TO_SHARE)

