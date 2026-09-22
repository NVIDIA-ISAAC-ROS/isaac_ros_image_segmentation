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

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_segment_anything/segment_anything_binarize_tensor.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

namespace
{
constexpr int32_t kExpectedChannelCount = 1;

// DLPack DLDataTypeCode values used by this node (see ExperimentalTensor.msg).
constexpr uint8_t kDLUInt = 1;
constexpr uint8_t kDLFloat = 2;
}  // namespace

SegmentAnythingDecoderNode::SegmentAnythingDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("segment_anything_decoder", options),
  mask_width_(declare_parameter<int16_t>("mask_width", 960)),
  mask_height_(declare_parameter<int16_t>("mask_height", 544)),
  max_batch_size_(declare_parameter<int16_t>("max_batch_size", 20)),
  tensor_name_(declare_parameter<std::string>("tensor_name", ""))
{
  // Validate parameters: int16_t allows non-positive values, which would
  // either overflow to SIZE_MAX-ish after static_cast (negatives) or be
  // rejected later with a generic error (zero).
  if (max_batch_size_ <= 0 || mask_height_ <= 0 || mask_width_ <= 0) {
    RCLCPP_ERROR(
      get_logger(),
      "Invalid mask parameters: max_batch_size=%d, mask_height=%d, "
      "mask_width=%d (all must be > 0)",
      max_batch_size_, mask_height_, mask_width_);
    throw std::invalid_argument(
            "segment_anything_decoder: max_batch_size, mask_height, and "
            "mask_width must all be > 0");
  }

  cudaStreamCreate(&cuda_stream_);

  // Initialize publisher
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  output_pub_ = create_publisher<TensorList>(
    "segment_anything/raw_segmentation_mask", rclcpp::QoS(1), pub_options);

  // Initialize subscriber
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  input_sub_ = create_subscription<TensorList>(
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
  const TensorList::ConstSharedPtr & msg)
{
  // Get the tensor by name, or fall back to the first tensor
  if (msg->tensors.empty()) {
    RCLCPP_ERROR(get_logger(), "Input tensor list is empty");
    return;
  }
  const tensor_msgs::msg::ExperimentalTensor * named_tensor = nullptr;
  if (!tensor_name_.empty()) {
    // Tensor names live in the TensorList-level names array, parallel to tensors.
    for (size_t i = 0; i < msg->names.size() && i < msg->tensors.size(); ++i) {
      if (msg->names[i] == tensor_name_) {
        named_tensor = &msg->tensors[i];
        break;
      }
    }
    if (!named_tensor) {
      RCLCPP_ERROR(get_logger(), "Tensor '%s' not found in input list", tensor_name_.c_str());
      return;
    }
  }
  const tensor_msgs::msg::ExperimentalTensor & tensor =
    named_tensor ? *named_tensor : msg->tensors.at(0);

  // Validate shape: expect NCHW with channels == 1
  if (tensor.shape.size() != 4) {
    RCLCPP_ERROR(
      get_logger(), "Expected 4D tensor (NCHW), got rank %zu", tensor.shape.size());
    return;
  }

  // The mask kernel reads the input as float32.
  if (tensor.dtype_code != kDLFloat || tensor.dtype_bits != 32 || tensor.dtype_lanes != 1) {
    RCLCPP_ERROR(
      get_logger(),
      "Expected float32 input tensor, got dtype_code=%u dtype_bits=%u dtype_lanes=%u",
      static_cast<unsigned int>(tensor.dtype_code),
      static_cast<unsigned int>(tensor.dtype_bits),
      static_cast<unsigned int>(tensor.dtype_lanes));
    return;
  }

  int32_t batch_size = static_cast<int32_t>(tensor.shape[0]);
  int32_t channels = static_cast<int32_t>(tensor.shape[1]);
  int32_t height = static_cast<int32_t>(tensor.shape[2]);
  int32_t width = static_cast<int32_t>(tensor.shape[3]);

  if (channels != kExpectedChannelCount) {
    RCLCPP_ERROR(
      get_logger(),
      "Expected %d channel(s), got %d", kExpectedChannelCount, channels);
    return;
  }

  // Sanity-cap the frame against the declared maximum mask dimensions and
  // drop anything larger to guard against malformed input.
  if (batch_size > max_batch_size_ ||
    height > mask_height_ ||
    width > mask_width_)
  {
    RCLCPP_ERROR(
      get_logger(),
      "Input tensor exceeds declared bounds: got [%d, %d, %d, %d], "
      "max [%d, %d, %d, %d] (NCHW)",
      batch_size, channels, height, width,
      max_batch_size_, kExpectedChannelCount, mask_height_, mask_width_);
    return;
  }

  // Total number of elements
  size_t num_elements = batch_size * channels * height * width;

  // Allocate an output mask buffer backed by the cuda_buffer backend. Strides
  // are left empty (the DLPack convention for contiguous row-major).
  tensor_msgs::msg::ExperimentalTensor output_tensor;
  output_tensor.dtype_code = kDLUInt;
  output_tensor.dtype_bits = 8;
  output_tensor.dtype_lanes = 1;
  output_tensor.shape = {batch_size, channels, height, width};
  output_tensor.byte_offset = 0;
  // uint8: byte count == element count
  output_tensor.data = cuda_buffer_backend::allocate_buffer(num_elements);
  {
    // Hold the handles in named variables so they outlive the kernel: the
    // ReadHandle destructor records the read-completion event, which must
    // happen after the read, not before. cuda_stream_ is the consumer's own
    // stream so it waits on the producer's event before the kernel runs.
    auto output_write_handle =
      cuda_buffer_backend::from_output_buffer(output_tensor.data, cuda_stream_);
    auto input_read_handle =
      cuda_buffer_backend::from_input_buffer(tensor.data, cuda_stream_);
    const float * input_data = reinterpret_cast<const float *>(
      input_read_handle.get_ptr() + tensor.byte_offset);
    ThresholdFloatToUint8OnGPU(
      input_data, output_write_handle.get_ptr(), num_elements, cuda_stream_);

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(), "CUDA kernel error: %s", cudaGetErrorString(kernel_err));
      return;
    }
  }  // WriteHandle destructor records the producer-side CUDA event here.

  std_msgs::msg::Header header = msg->header;

  isaac_ros_tensor_msgs::msg::TensorList out_msg;
  out_msg.header = header;
  // Unnamed, matching the pre-migration output; the downstream tensor-to-image
  // node takes the first tensor. names stays parallel to tensors.
  out_msg.names.push_back("");
  out_msg.tensors.push_back(std::move(output_tensor));

  output_pub_->publish(std::move(out_msg));
}

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode)
