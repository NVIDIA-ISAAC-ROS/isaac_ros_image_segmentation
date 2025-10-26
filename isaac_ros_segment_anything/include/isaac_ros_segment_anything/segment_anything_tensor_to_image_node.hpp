// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <memory>
#include <string>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

/**
 * @brief Node that converts a TensorList containing segmentation masks to a binary mask image.
 *
 * This node subscribes to a NITROS TensorList topic and converts the segmentation masks to a
 * binary mask image using NITROS. It only supports batch size 1 tensors.
 */
class TensorToImageNode : public rclcpp::Node
{
public:
  /**
   * @brief Constructor for TensorListToBinaryMaskNode.
   * @param options The node options.
   */
  explicit TensorToImageNode(const rclcpp::NodeOptions & options);

  /**
   * @brief Destructor for TensorToImageNode.
   */
  ~TensorToImageNode() override = default;

private:
  /**
   * @brief Callback for processing incoming TensorList messages.
   * @param tensor_list The received TensorList message.
   */
  void TensorListCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & tensor_list);

  // QoS settings
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  // NITROS subscribers and publishers
  std::shared_ptr<
    nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>> tensor_list_sub_;

  std::shared_ptr<
    nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> binary_mask_pub_;

  // Standard ROS publisher for bounding boxes
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;

  // CUDA stream for GPU operations
  cudaStream_t stream_;
};

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia
