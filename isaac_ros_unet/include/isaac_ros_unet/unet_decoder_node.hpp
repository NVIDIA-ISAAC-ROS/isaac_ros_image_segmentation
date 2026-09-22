// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_UNET__UNET_DECODER_NODE_HPP_
#define ISAAC_ROS_UNET__UNET_DECODER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "isaac_ros_unet_kernels/segmentation_mask_colorizer.cu.hpp"
#include "isaac_ros_unet_kernels/segmentation_postprocessor.cu.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace unet
{

class UNetDecoderNode : public rclcpp::Node
{
public:
  explicit UNetDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~UNetDecoderNode();

private:
  void TensorCallback(
    const isaac_ros_tensor_msgs::msg::TensorList::ConstSharedPtr & msg);

  // Parameters
  std::string color_segmentation_mask_encoding_;
  std::vector<int64_t> color_palette_;
  std::string network_output_type_;
  std::string data_format_;

  // Parsed enum values
  nvidia::isaac_ros::NetworkOutputType network_output_type_value_;
  nvidia::isaac_ros::DataFormat data_format_value_;
  nvidia::isaac_ros::ColorImageEncodings color_encoding_value_;

  // ROS 2 pub/sub
  rclcpp::Subscription<isaac_ros_tensor_msgs::msg::TensorList>::SharedPtr tensor_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr colored_mask_pub_;

  // CUDA resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::ArrayView<int64_t> color_palette_device_;
};

}  // namespace unet
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_UNET__UNET_DECODER_NODE_HPP_
