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

#ifndef ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DECODER_NODE_HPP_
#define ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DECODER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

class SegmentAnythingDecoderNode : public nitros::NitrosNode
{
public:
  explicit SegmentAnythingDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~SegmentAnythingDecoderNode();

  void postLoadGraphCallback() override;

private:
  int16_t mask_width_;

  // The height of the segmentation mask
  int16_t mask_height_;

  // Needed to calculate block size. It is max batch size for prompt bboxes
  int16_t max_batch_size_;
};

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DECODER_NODE_HPP_
