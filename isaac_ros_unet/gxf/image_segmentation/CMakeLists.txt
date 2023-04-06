# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(gxf_segmentation_postprocessor LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

# Dependencies
find_package(CUDAToolkit)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    core
    cuda
    multimedia
    serialization
    std
)
include(YamlCpp)

# Create extension
add_library(gxf_segmentation_postprocessor SHARED
    segmentation_mask_colorizer.cpp
    segmentation_mask_colorizer.hpp
    segmentation_postprocessing_utils.hpp
    segmentation_postprocessor_ext.cpp
    segmentation_postprocessor.cpp
    segmentation_postprocessor.hpp
    segmentation_postprocessor.cu.cpp
    segmentation_postprocessor.cu.hpp
    segmentation_mask_colorizer.cu.cpp
    segmentation_mask_colorizer.cu.hpp
)

# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  segmentation_postprocessor.cu.cpp
  segmentation_postprocessor.cu.hpp
  segmentation_mask_colorizer.cu.cpp
  segmentation_mask_colorizer.cu.hpp
  PROPERTIES LANGUAGE CUDA
)

target_link_libraries(gxf_segmentation_postprocessor
  PUBLIC
    CUDA::cudart
    GXF::cuda
    GXF::multimedia
    GXF::std
    yaml-cpp
)
