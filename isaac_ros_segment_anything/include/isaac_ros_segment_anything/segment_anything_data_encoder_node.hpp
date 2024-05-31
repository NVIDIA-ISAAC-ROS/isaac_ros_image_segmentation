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

#ifndef ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DATA_ENCODER_NODE_HPP_
#define ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DATA_ENCODER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "isaac_ros_nitros_detection2_d_array_type/nitros_detection2_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

class SegmentAnythingDataEncoderNode : public nitros::NitrosNode
{
public:
  explicit SegmentAnythingDataEncoderNode(
    const rclcpp::NodeOptions options = rclcpp::NodeOptions()
  );
  ~SegmentAnythingDataEncoderNode();

  void postLoadGraphCallback() override;

private:
  // Needed to calculate block size. It is max batch size for prompt bboxes
  int32_t max_batch_size_;

  // Prompt type
  std::string prompt_input_type_;
  bool has_input_mask_;
  std::vector<int64_t> orig_img_dims_;
};

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DATA_ENCODER_NODE_HPP_
