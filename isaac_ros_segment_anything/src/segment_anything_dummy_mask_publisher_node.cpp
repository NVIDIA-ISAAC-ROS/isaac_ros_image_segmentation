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

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "isaac_ros_segment_anything/segment_anything_dummy_mask_publisher_node.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"
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
  sub_options.acceptable_buffer_backends = "any";
  tensor_sub_ = create_subscription<isaac_ros_tensor_msgs::msg::TensorList>(
    "tensor_pub", rclcpp::QoS(10),
    std::bind(&DummyMaskPublisher::InputCallback, this, std::placeholders::_1),
    sub_options);

  // Initialize publisher
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor_pub_ = create_publisher<isaac_ros_tensor_msgs::msg::TensorList>(
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
  const isaac_ros_tensor_msgs::msg::TensorList::ConstSharedPtr & msg)
{
  constexpr size_t kBufferSize = 256 * 256 * 4;

  std_msgs::msg::Header header;
  header.stamp.sec = msg->header.stamp.sec;
  header.stamp.nanosec = msg->header.stamp.nanosec;
  header.frame_id = msg->header.frame_id;

  auto out = std::make_unique<isaac_ros_tensor_msgs::msg::TensorList>();
  out->header = header;
  tensor_msgs::msg::ExperimentalTensor t;
  t.dtype_code = 2;   // DLPack Float
  t.dtype_bits = 32;
  t.dtype_lanes = 1;
  t.shape = {1, 1, 256, 256};
  // strides left empty: contiguous row-major per DLPack convention
  t.byte_offset = 0;
  t.data = cuda_buffer_backend::allocate_buffer(kBufferSize);
  {
    auto wh = cuda_buffer_backend::from_output_buffer(t.data, stream_);
    CHECK_CUDA_ERROR(
      cudaMemsetAsync(wh.get_ptr(), 0, kBufferSize, stream_),
      "Failed to zero GPU memory");
  }
  // Tensor names live in the TensorList-level names array, parallel to tensors.
  out->names.push_back(tensor_name_);
  out->tensors.push_back(std::move(t));
  tensor_pub_->publish(std::move(out));
}

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::segment_anything::DummyMaskPublisher)
