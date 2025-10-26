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

#include <cuda_runtime.h>
#include <string>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_segment_anything/segment_anything_dummy_mask_publisher_node.hpp"
namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

DummyMaskPublisher::DummyMaskPublisher(const rclcpp::NodeOptions options)
: rclcpp::Node("dummy_mask_publisher", options),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_pub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&DummyMaskPublisher::InputCallback, this,
      std::placeholders::_1))},
  nitros_pub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosTensorList>>(
      this, "mask",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "input_mask")}
{
  // Initialize CUDA stream
  CHECK_CUDA_ERROR(
    ::nvidia::isaac_ros::common::initNamedCudaStream(
      stream_, "isaac_ros_dummy_mask_publisher_node"),
    "Error initializing CUDA stream");
}

DummyMaskPublisher::~DummyMaskPublisher() = default;

void DummyMaskPublisher::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  // Buffer size for mask
  size_t buffer_size{256 * 256 * 4};

  // Allocate CUDA buffer in the stream to not block the main default stream and other work.
  void * buffer;
  CHECK_CUDA_ERROR(
    cudaMallocAsync(&buffer, buffer_size, stream_),
    "Failed to allocate GPU memory");
  CHECK_CUDA_ERROR(
    cudaMemsetAsync(buffer, 0, buffer_size, stream_),
    "Failed to zero GPU memory");

  // Adding header data
  std_msgs::msg::Header header;
  header.stamp.sec = msg.GetTimestampSeconds();
  header.stamp.nanosec = msg.GetTimestampNanoseconds();
  header.frame_id = msg.GetFrameId();

  // Sync the stream
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "Failed to synchronize CUDA stream");

  // Create tensor list with tensor wrapping CUDA buffer
  nvidia::isaac_ros::nitros::NitrosTensorList tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(
    tensor_name_,
    (
      nvidia::isaac_ros::nitros::NitrosTensorBuilder()
      .WithShape({1, 1, 256, 256})
      .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
      .WithData(buffer)
      .Build()
    )
    )
    .Build();
  nitros_pub_->publish(tensor_list);
}

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::segment_anything::DummyMaskPublisher)
