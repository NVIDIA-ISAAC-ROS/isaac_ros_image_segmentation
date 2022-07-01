/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

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
