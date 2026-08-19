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

#include <cuda_runtime.h>
#include <string>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_shape.hpp"
#include "isaac_ros_segment_anything/segment_anything_dummy_mask_publisher_node.hpp"
namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

DummyMaskPublisher::DummyMaskPublisher(const rclcpp::NodeOptions options)
: rclcpp::Node("dummy_mask_publisher", options),
  tensor_name_{declare_parameter<std::string>("tensor_name", "input_mask")}
{
  // Initialize subscriber
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  nitros_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "tensor_pub", rclcpp::QoS(10),
    std::bind(&DummyMaskPublisher::InputCallback, this, std::placeholders::_1),
    sub_options);

  // Initialize publisher
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  nitros_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "mask", rclcpp::QoS(10), pub_options);

  // Initialize CUDA stream
  CHECK_CUDA_ERROR(
    ::nvidia::isaac_ros::common::initNamedCudaStream(
      stream_, "isaac_ros_dummy_mask_publisher_node"),
    "Error initializing CUDA stream");
}

DummyMaskPublisher::~DummyMaskPublisher()
{
  if (stream_ != nullptr) {
    const cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to destroy CUDA stream: %s",
        cudaGetErrorString(err));
    }
    stream_ = nullptr;
  }
}

void DummyMaskPublisher::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg)
{
  constexpr size_t kBufferSize = 256 * 256 * 4;

  void * buffer = nullptr;
  CHECK_CUDA_ERROR(
    cudaMallocAsync(&buffer, kBufferSize, stream_),
    "Failed to allocate GPU memory");

  nvidia::isaac_ros::nitros::NitrosTensor mask_tensor;
  {
    // Use from_external's default sync cudaFree deleter: capturing the node's
    // stream_ in an async deleter would dangle if a published tensor outlives
    // ~DummyMaskPublisher (which destroys the stream).
    auto write_handle = mask_tensor.from_external(
      tensor_name_, buffer, kBufferSize,
      nvidia::isaac_ros::nitros::NitrosTensorShape({1, 1, 256, 256}),
      nvidia::isaac_ros::nitros::NitrosDataType::kFloat32, stream_);
    CHECK_CUDA_ERROR(
      cudaMemsetAsync(write_handle.get_ptr(), 0, kBufferSize, stream_),
      "Failed to zero GPU memory");
  }  // WriteHandle destructor records the producer-side CUDA event.

  std_msgs::msg::Header header;
  header.stamp.sec = static_cast<int32_t>(msg->get_timestamp_sec());
  header.stamp.nanosec = msg->get_timestamp_nsec();
  header.frame_id = msg->get_frame_id();

  nvidia::isaac_ros::nitros::NitrosTensorList tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(tensor_name_, std::move(mask_tensor))
    .Build();
  nitros_pub_->publish(std::move(tensor_list));
}

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::segment_anything::DummyMaskPublisher)
