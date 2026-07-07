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

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_unet_kernels/segmentation_postprocessor.cu.hpp"
#include "isaac_ros_unet_kernels/segmentation_mask_colorizer.cu.hpp"

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
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg);

  // Parameters
  std::string color_segmentation_mask_encoding_;
  std::vector<int64_t> color_palette_;
  std::string network_output_type_;
  std::string data_format_;
  int16_t mask_width_;
  int16_t mask_height_;

  // Parsed enum values
  nvidia::isaac_ros::NetworkOutputType network_output_type_value_;
  nvidia::isaac_ros::DataFormat data_format_value_;
  nvidia::isaac_ros::ColorImageEncodings color_encoding_value_;

  // ROS 2 pub/sub
  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr tensor_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr raw_mask_pub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr colored_mask_pub_;

  // CUDA resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;
  nvidia::isaac_ros::ArrayView<int64_t> color_palette_device_;
};

}  // namespace unet
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_UNET__UNET_DECODER_NODE_HPP_
