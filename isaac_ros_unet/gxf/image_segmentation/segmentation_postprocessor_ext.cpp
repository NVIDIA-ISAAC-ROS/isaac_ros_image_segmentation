// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
#include "segmentation_mask_colorizer.hpp"
#include "segmentation_postprocessor.hpp"

#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x15ed574714ef4f56, 0xb277090d5b35a4ff,
                         "SegmentationPostprocessorExtension",
                         "Isaac ROS Segmentation PostProcessor Extension", "NVIDIA", "0.0.1",
                         "LICENSE");

GXF_EXT_FACTORY_ADD(0xe9681b9e1b864649, 0x8fc86530f45f9d09,
                    nvidia::isaac_ros::SegmentationPostprocessor, nvidia::gxf::Codelet,
                    "Generates a raw segmentation mask from a tensor");

GXF_EXT_FACTORY_ADD(0xfbdd0d490ccc4df7, 0xba990847538359b9,
                    nvidia::isaac_ros::SegmentationMaskColorizer, nvidia::gxf::Codelet,
                    "Mask generation codelet");
GXF_EXT_FACTORY_END()
