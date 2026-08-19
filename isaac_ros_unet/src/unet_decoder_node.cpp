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

#include "isaac_ros_unet/unet_decoder_node.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "isaac_ros_common/qos.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace unet
{
namespace
{

bool IsSupportedNetworkOutputType(const std::string & network_output_type)
{
  return network_output_type == "softmax" ||
         network_output_type == "sigmoid" ||
         network_output_type == "argmax";
}

bool IsSupportedDataFormat(const std::string & data_format)
{
  return data_format == "NCHW" ||
         data_format == "HWC" ||
         data_format == "NHWC";
}

NetworkOutputType ParseNetworkOutputType(const std::string & name)
{
  if (name == "sigmoid") {return NetworkOutputType::kSigmoid;}
  if (name == "softmax") {return NetworkOutputType::kSoftmax;}
  if (name == "argmax") {return NetworkOutputType::kArgmax;}
  throw std::invalid_argument("Unsupported network output type: " + name);
}

DataFormat ParseDataFormat(const std::string & name)
{
  if (name == "NCHW") {return DataFormat::kNCHW;}
  if (name == "HWC") {return DataFormat::kHWC;}
  if (name == "NHWC") {return DataFormat::kNHWC;}
  throw std::invalid_argument("Unsupported data format: " + name);
}

Shape ExtractShape(
  const nvidia::isaac_ros::nitros::NitrosTensor & tensor,
  DataFormat data_format)
{
  Shape shape{};
  auto dims = tensor.shape().dims();
  const char * format_name = nullptr;
  size_t required_dims = 0;
  switch (data_format) {
    case DataFormat::kHWC:
      format_name = "HWC";
      required_dims = 3;
      break;
    case DataFormat::kNCHW:
      format_name = "NCHW";
      required_dims = 4;
      break;
    case DataFormat::kNHWC:
      format_name = "NHWC";
      required_dims = 4;
      break;
    default:
      throw std::invalid_argument(
              "ExtractShape received unknown DataFormat value: " +
              std::to_string(static_cast<int>(data_format)));
  }
  if (dims.size() < required_dims) {
    throw std::invalid_argument(
            "NitrosTensor has insufficient dimensions for DataFormat::" +
            std::string(format_name) + ": expected at least " +
            std::to_string(required_dims) + ", got " + std::to_string(dims.size()));
  }
  switch (data_format) {
    case DataFormat::kHWC:
      shape.height = dims[0];
      shape.width = dims[1];
      shape.channels = dims[2];
      break;
    case DataFormat::kNCHW:
      shape.channels = dims[1];
      shape.height = dims[2];
      shape.width = dims[3];
      break;
    case DataFormat::kNHWC:
      shape.height = dims[1];
      shape.width = dims[2];
      shape.channels = dims[3];
      break;
    default:
      throw std::invalid_argument(
              "ExtractShape received unknown DataFormat value: " +
              std::to_string(static_cast<int>(data_format)));
  }
  return shape;
}

}  // namespace

UNetDecoderNode::UNetDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("unet_decoder_node", options),
  color_segmentation_mask_encoding_(
    declare_parameter<std::string>("color_segmentation_mask_encoding", "rgb8")),
  color_palette_(
    declare_parameter<std::vector<int64_t>>("color_palette", std::vector<int64_t>({}))),
  network_output_type_(declare_parameter<std::string>("network_output_type", "softmax")),
  data_format_(declare_parameter<std::string>("data_format", "NHWC")),
  mask_width_(declare_parameter<int16_t>("mask_width", 960)),
  mask_height_(declare_parameter<int16_t>("mask_height", 544))
{
  // Validate color_segmentation_mask_encoding
  if (color_segmentation_mask_encoding_.empty()) {
    RCLCPP_ERROR(get_logger(), "Received empty color segmentation mask encoding!");
    throw std::invalid_argument("Received empty color segmentation mask encoding!");
  }
  if (color_segmentation_mask_encoding_ != sensor_msgs::image_encodings::RGB8 &&
    color_segmentation_mask_encoding_ != sensor_msgs::image_encodings::BGR8)
  {
    RCLCPP_ERROR(
      get_logger(), "Received invalid color segmentation mask encoding: %s",
      color_segmentation_mask_encoding_.c_str());
    throw std::invalid_argument(
            "Received invalid color segmentation mask encoding: " +
            color_segmentation_mask_encoding_);
  }

  // Validate color palette
  if (color_palette_.empty()) {
    RCLCPP_ERROR(
      get_logger(),
      "Received empty color palette! Fill this with a 24-bit hex color for each class!");
    throw std::invalid_argument(
            "Received empty color palette! Fill this with a 24-bit hex color for each class!");
  }

  // Validate network output type
  if (!IsSupportedNetworkOutputType(network_output_type_)) {
    RCLCPP_ERROR(
      get_logger(), "Received invalid network output type: %s!",
      network_output_type_.c_str());
    throw std::invalid_argument("Received invalid network output type: " + network_output_type_);
  }

  // Validate data format
  if (!IsSupportedDataFormat(data_format_)) {
    RCLCPP_ERROR(
      get_logger(), "Received invalid data format: %s!",
      data_format_.c_str());
    throw std::invalid_argument("Received invalid data format: " + data_format_);
  }

  // Parse enums
  network_output_type_value_ = ParseNetworkOutputType(network_output_type_);
  data_format_value_ = ParseDataFormat(data_format_);
  color_encoding_value_ = (color_segmentation_mask_encoding_ ==
    sensor_msgs::image_encodings::RGB8) ?
    ColorImageEncodings::kRGB8 :
    ColorImageEncodings::kBGR8;

  // Create CUDA stream
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("unet_decoder");

  // Create CUDA memory pool
  // Raw mask: mono8 (1 byte/pixel), Colored mask: rgb8/bgr8 (3 bytes/pixel)
  // Pool block size should accommodate the larger of the two
  size_t color_block_size = static_cast<size_t>(mask_width_) * mask_height_ * 3;
  cudaError_t err = pool_.create(
    color_block_size,
    40,  // num_blocks
    nitros::CUDAMemoryPool::MemoryType::Device);
  CHECK_CUDA_ERROR(err, "Failed to create CUDA memory pool");

  // Copy color palette to device
  int64_t * palette_data{nullptr};
  CHECK_CUDA_ERROR(
    cudaMallocAsync(&palette_data, sizeof(int64_t) * color_palette_.size(), *cuda_stream_),
    "Failed to allocate device color palette");
  color_palette_device_.data.reset(palette_data);
  color_palette_device_.size = color_palette_.size();
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      color_palette_device_.data.get(), color_palette_.data(),
      color_palette_.size() * sizeof(int64_t), cudaMemcpyHostToDevice, *cuda_stream_),
    "Failed to copy color palette to device");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_), "Failed to sync stream");

  // Create subscriber and publishers
  rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos");
  rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "tensor_sub", input_qos,
    std::bind(&UNetDecoderNode::TensorCallback, this, std::placeholders::_1),
    sub_options);

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  raw_mask_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    "unet/raw_segmentation_mask", output_qos, pub_options);
  colored_mask_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    "unet/colored_segmentation_mask", output_qos, pub_options);
}

void UNetDecoderNode::TensorCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg)
{
  if (msg->num_tensors() == 0) {
    RCLCPP_ERROR(get_logger(), "Received empty tensor list!");
    return;
  }

  const auto & tensor = msg->get_tensor(0);
  Shape shape = ExtractShape(tensor, data_format_value_);

  if (shape.channels > kMaxChannelCount) {
    RCLCPP_ERROR(
      get_logger(), "Received %d channels, max is %ld",
      shape.channels, kMaxChannelCount);
    return;
  }

  // Get input tensor GPU data
  auto tensor_read_handle = tensor.get_read_handle(*cuda_stream_);
  const uint8_t * tensor_gpu_ptr = tensor_read_handle.get_ptr();

  // Allocate raw mask output (mono8)
  auto raw_mask_msg = std::make_unique<nvidia::isaac_ros::nitros::NitrosImage>();
  auto colored_mask_msg = std::make_unique<nvidia::isaac_ros::nitros::NitrosImage>();

  {
    size_t raw_step = static_cast<size_t>(shape.width);
    auto raw_write_handle = raw_mask_msg->from_pool(
      pool_, shape.width, shape.height, raw_step,
      sensor_msgs::image_encodings::MONO8, *cuda_stream_);
    uint8_t * raw_mask_ptr = raw_write_handle.get_ptr();

    // Run postprocessing kernel
    if (network_output_type_value_ == NetworkOutputType::kArgmax) {
      auto data_type = tensor.data_type();
      if (data_type == nitros::NitrosDataType::kInt32) {
        CopyTensorData<int32_t>(
          network_output_type_value_, data_format_value_, shape,
          reinterpret_cast<const int32_t *>(tensor_gpu_ptr),
          raw_mask_ptr, *cuda_stream_);
      } else if (data_type == nitros::NitrosDataType::kInt64) {
        CopyTensorData<int64_t>(
          network_output_type_value_, data_format_value_, shape,
          reinterpret_cast<const int64_t *>(tensor_gpu_ptr),
          raw_mask_ptr, *cuda_stream_);
      } else {
        RCLCPP_ERROR(get_logger(), "Unsupported tensor element type for argmax.");
        return;
      }
    } else {
      cuda_postprocess(
        network_output_type_value_, data_format_value_, shape,
        reinterpret_cast<const float *>(tensor_gpu_ptr),
        raw_mask_ptr, *cuda_stream_);
    }

    // Check for CUDA errors
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(), "Postprocessing kernel error: %s",
        cudaGetErrorString(kernel_err));
      return;
    }

    // Allocate colored mask output (rgb8 or bgr8)
    size_t color_step = static_cast<size_t>(shape.width) * 3;
    auto color_write_handle = colored_mask_msg->from_pool(
      pool_, shape.width, shape.height, color_step,
      color_segmentation_mask_encoding_, *cuda_stream_);
    uint8_t * colored_mask_ptr = color_write_handle.get_ptr();

    // Run colorization kernel
    ColorizeSegmentationMask(
      colored_mask_ptr, shape.width, shape.height,
      color_encoding_value_, raw_mask_ptr,
      color_palette_device_, *cuda_stream_);

    kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(), "Colorization kernel error: %s",
        cudaGetErrorString(kernel_err));
      return;
    }

    // Sync stream before publishing
    CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_), "Failed to sync stream");
  }

  // Copy metadata from input
  raw_mask_msg->timestamp_sec = msg->get_timestamp_sec();
  raw_mask_msg->timestamp_nsec = msg->get_timestamp_nsec();
  raw_mask_msg->frame_id = msg->get_frame_id();

  colored_mask_msg->timestamp_sec = msg->get_timestamp_sec();
  colored_mask_msg->timestamp_nsec = msg->get_timestamp_nsec();
  colored_mask_msg->frame_id = msg->get_frame_id();

  // Publish both outputs
  raw_mask_pub_->publish(std::move(raw_mask_msg));
  colored_mask_pub_->publish(std::move(colored_mask_msg));
}

UNetDecoderNode::~UNetDecoderNode()
{
  // color_palette_device_.data was allocated via cudaMallocAsync on cuda_stream_,
  // but ArrayView's unique_ptr deleter is cudaFree (sync). Release ownership and
  // free with cudaFreeAsync on the same stream to keep stream-ordered semantics.
  if (color_palette_device_.data && cuda_stream_) {
    int64_t * raw = color_palette_device_.data.release();
    cudaFreeAsync(raw, *cuda_stream_);
  }
}

}  // namespace unet
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::unet::UNetDecoderNode)
