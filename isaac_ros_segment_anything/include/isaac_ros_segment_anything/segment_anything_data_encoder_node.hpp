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

#ifndef ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DATA_ENCODER_NODE_HPP_
#define ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DATA_ENCODER_NODE_HPP_

#include <cuda_runtime.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

class SegmentAnythingDataEncoderNode : public rclcpp::Node
{
public:
  explicit SegmentAnythingDataEncoderNode(
    const rclcpp::NodeOptions options = rclcpp::NodeOptions()
  );
  ~SegmentAnythingDataEncoderNode();

private:
  using NitrosTensorList = nvidia::isaac_ros::nitros::NitrosTensorList;
  using Detection2DArray = vision_msgs::msg::Detection2DArray;

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    Detection2DArray, NitrosTensorList, NitrosTensorList>;
  using ExactSync = message_filters::Synchronizer<ExactPolicy>;

  void SyncCallback(
    const Detection2DArray::ConstSharedPtr & prompts,
    const NitrosTensorList::ConstSharedPtr & image_tensor,
    const NitrosTensorList::ConstSharedPtr & mask_tensor);

  void DetectionToSAMPrompt(
    const std::vector<vision_msgs::msg::Detection2D> & detections,
    std::vector<float> & prompt_vec,
    std::vector<float> & label_vec);

  // Subscribers (message_filters)
  message_filters::Subscriber<Detection2DArray> prompt_sub_;
  message_filters::Subscriber<NitrosTensorList> image_sub_;
  message_filters::Subscriber<NitrosTensorList> mask_sub_;
  std::shared_ptr<ExactSync> exact_sync_;

  // Publisher
  rclcpp::Publisher<NitrosTensorList>::SharedPtr output_pub_;

  // Parameters
  int32_t max_batch_size_;
  std::string prompt_input_type_;
  bool has_input_mask_;
  std::vector<int64_t> orig_img_dims_;

  // Precomputed values
  uint16_t resized_width_;
  uint16_t resized_height_;
  bool is_bbox_prompt_;

  static constexpr uint16_t kImageWidth = 1024;
  static constexpr uint16_t kImageHeight = 1024;
  static constexpr uint32_t kNumPointsPerBbox = 2;

  // CUDA stream
  cudaStream_t cuda_stream_{};
};

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_SEGMENT_ANYTHING__SEGMENT_ANYTHING_DATA_ENCODER_NODE_HPP_
