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

#include "isaac_ros_segment_anything/segment_anything_decoder_node.hpp"

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_shape.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_data_type.hpp"
#include "isaac_ros_segment_anything/segment_anything_binarize_tensor.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

namespace
{
constexpr int32_t kExpectedChannelCount = 1;
}  // namespace

SegmentAnythingDecoderNode::SegmentAnythingDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("segment_anything_decoder", options),
  mask_width_(declare_parameter<int16_t>("mask_width", 960)),
  mask_height_(declare_parameter<int16_t>("mask_height", 544)),
  max_batch_size_(declare_parameter<int16_t>("max_batch_size", 20)),
  tensor_name_(declare_parameter<std::string>("tensor_name", ""))
{
  cudaStreamCreate(&cuda_stream_);

  // Initialize publisher
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  output_pub_ = create_publisher<NitrosTensorList>(
    "segment_anything/raw_segmentation_mask", rclcpp::QoS(1), pub_options);

  // Initialize subscriber
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  input_sub_ = create_subscription<NitrosTensorList>(
    "tensor_sub", rclcpp::QoS(1),
    std::bind(&SegmentAnythingDecoderNode::InputCallback, this, std::placeholders::_1),
    sub_options);
}

SegmentAnythingDecoderNode::~SegmentAnythingDecoderNode()
{
  if (cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
  }
}

void SegmentAnythingDecoderNode::InputCallback(
  const NitrosTensorList::ConstSharedPtr & msg)
{
  // Get the tensor by name, or fall back to the first tensor
  if (msg->get_tensors().empty()) {
    RCLCPP_ERROR(get_logger(), "Input tensor list is empty");
    return;
  }
  std::shared_ptr<nitros::NitrosTensor> named_tensor;
  if (!tensor_name_.empty()) {
    named_tensor = msg->get_tensor_by_name(tensor_name_);
    if (!named_tensor) {
      RCLCPP_ERROR(get_logger(), "Tensor '%s' not found in input list", tensor_name_.c_str());
      return;
    }
  }
  const nitros::NitrosTensor & tensor =
    named_tensor ? *named_tensor : msg->get_tensors().at(0);

  // Validate shape: expect NCHW with channels == 1
  if (tensor.shape().rank() != 4) {
    RCLCPP_ERROR(
      get_logger(), "Expected 4D tensor (NCHW), got rank %u", tensor.shape().rank());
    return;
  }

  int32_t batch_size = tensor.shape().dims()[0];
  int32_t channels = tensor.shape().dims()[1];
  int32_t height = tensor.shape().dims()[2];
  int32_t width = tensor.shape().dims()[3];

  if (channels != kExpectedChannelCount) {
    RCLCPP_ERROR(
      get_logger(),
      "Expected %d channel(s), got %d", kExpectedChannelCount, channels);
    return;
  }

  // Total number of elements
  size_t num_elements = batch_size * channels * height * width;

  // Allocate output buffer (uint8) on GPU
  void * output_gpu = nullptr;
  cudaMalloc(&output_gpu, num_elements * sizeof(uint8_t));

  // Run threshold kernel: float > 0 → 1, else → 0
  const float * input_data = reinterpret_cast<const float *>(
    tensor.get_read_handle(msg->get_stream()).get_ptr());
  ThresholdFloatToUint8OnGPU(
    input_data, static_cast<uint8_t *>(output_gpu), num_elements, cuda_stream_);

  cudaError_t kernel_err = cudaGetLastError();
  if (kernel_err != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(), "CUDA kernel error: %s", cudaGetErrorString(kernel_err));
    cudaFree(output_gpu);
    return;
  }

  cudaStreamSynchronize(cuda_stream_);

  // Build output tensor list
  std_msgs::msg::Header header = msg->get_header();

  auto output_tensor_list = nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(
      "",
      nitros::NitrosTensorBuilder()
    .WithShape(nitros::NitrosTensorShape({batch_size, channels, height, width}))
    .WithDataType(nitros::NitrosDataType::kUnsigned8)
    .WithData(output_gpu)
    .WithReleaseCallback([output_gpu]() {cudaFree(output_gpu);})
    .Build())
    .Build();

  output_pub_->publish(std::move(output_tensor_list));
}

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode)
