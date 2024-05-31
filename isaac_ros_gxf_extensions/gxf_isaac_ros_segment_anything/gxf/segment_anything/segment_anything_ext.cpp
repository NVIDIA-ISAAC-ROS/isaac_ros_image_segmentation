// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "segment_anything_postprocessor.hpp"
#include "segment_anything_prompt_processor.hpp"
#include "segment_anything_msg_compositor.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xa3ed574714ef4f11, 0xc127090d5b35a477,
                         "SegmentAnythingExtension",
                         "Isaac ROS Segmentation PostProcessor Extension", "NVIDIA", "0.0.1",
                         "LICENSE");

GXF_EXT_FACTORY_ADD(0xe9681b9e1b864123, 0x8fc86530f45f9ab2,
                    nvidia::isaac_ros::SegmentAnythingPostprocessor, nvidia::gxf::Codelet,
                    "Generates a raw segmentation mask from a tensor");
GXF_EXT_FACTORY_ADD(0xe8211b9e1b864ab1, 0x2de86530f45f9cd3,
                    nvidia::isaac_ros::SegmentAnythingPromptProcessor, nvidia::gxf::Codelet,
                    "Transforms the input bboxes/points to SAM format.");
GXF_EXT_FACTORY_ADD(0xe12acb9e1b8642ba, 0x34c86170f32f9abc,
                    nvidia::isaac_ros::SegmentAnythingMsgCompositor, nvidia::gxf::Codelet,
                    "Composes a single msg with all the received tensors.");
GXF_EXT_FACTORY_ADD_0(
  0xa321601525594206, 0xbc12d9f22a134452,
  std::vector<nvidia::isaac_ros::Detection2D>,
  "Array of decoded 2D object detections in an image");
GXF_EXT_FACTORY_END()
