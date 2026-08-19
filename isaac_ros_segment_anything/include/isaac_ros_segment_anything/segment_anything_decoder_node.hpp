// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

class SegmentAnythingDecoderNode : public rclcpp::Node
{
public:
  explicit SegmentAnythingDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~SegmentAnythingDecoderNode();

private:
  using NitrosTensorList = nvidia::isaac_ros::nitros::NitrosTensorList;

  void InputCallback(const NitrosTensorList::ConstSharedPtr & msg);

  // Subscriber and publisher
  rclcpp::Subscription<NitrosTensorList>::SharedPtr input_sub_;
  rclcpp::Publisher<NitrosTensorList>::SharedPtr output_pub_;

  // Parameters
  int16_t mask_width_;
  int16_t mask_height_;
  int16_t max_batch_size_;
  std::string tensor_name_;

  // CUDA stream
  cudaStream_t cuda_stream_{};

  // Pre-allocated pool for output mask buffers; sized in the constructor by
  // max_batch_size_ * mask_height_ * mask_width_.
  nvidia::isaac_ros::nitros::CUDAMemoryPool output_pool_;
};

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DECODER_NODE_HPP_
