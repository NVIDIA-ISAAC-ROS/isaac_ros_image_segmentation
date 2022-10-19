// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace unet
{

class UNetDecoderNode : public nitros::NitrosNode
{
public:
  explicit UNetDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~UNetDecoderNode();

  void postLoadGraphCallback() override;

private:
  // The color encoding that the colored segmentation mask should be in
  // This should be either rgb8 or bgr8
  std::string color_segmentation_mask_encoding_;

  // The color palette for the color segmentation mask
  // There should be an element for each class
  // Note: only the first 24 bits are used
  std::vector<int64_t> color_palette_;

  // Whether sigmoid or softmax was performed by the network
  std::string network_output_type_;

  // The width of the segmentation mask
  int16_t mask_width_;

  // The height of the segmentation mask
  int16_t mask_height_;
};

}  // namespace unet
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_UNET__UNET_DECODER_NODE_HPP_
