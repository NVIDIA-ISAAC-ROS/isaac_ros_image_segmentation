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

#ifndef ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DUMMY_MASK_PUBLISHER_NODE_HPP_
#define ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DUMMY_MASK_PUBLISHER_NODE_HPP_

#include <memory>
#include <string>

#include "cuda_runtime.h"  // NOLINT - include .h without directory

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

class DummyMaskPublisher : public rclcpp::Node
{
public:
  explicit DummyMaskPublisher(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~DummyMaskPublisher();

private:
  void InputCallback(
    const isaac_ros_tensor_msgs::msg::TensorList::ConstSharedPtr & msg);

  // Subscription to input TensorList messages
  rclcpp::Subscription<isaac_ros_tensor_msgs::msg::TensorList>::SharedPtr tensor_sub_;

  // Publisher for output TensorList messages
  rclcpp::Publisher<isaac_ros_tensor_msgs::msg::TensorList>::SharedPtr tensor_pub_;

  // Name of tensor in TensorList
  std::string tensor_name_{};

  // CUDA stream for async operations
  cudaStream_t stream_{nullptr};
};

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DUMMY_MASK_PUBLISHER_NODE_HPP_
