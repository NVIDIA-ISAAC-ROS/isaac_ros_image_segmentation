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

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace unet
{
namespace
{

// DLPack DLDataTypeCode values, per tensor_msgs/ExperimentalTensor.msg. The element
// type is the triple {dtype_code, dtype_bits, dtype_lanes} rather than a single
// ordinal, so int32 is {kInt, 32, 1} and float32 is {kFloat, 32, 1}.
constexpr uint8_t kDLDataTypeCodeInt = 0;
constexpr uint8_t kDLDataTypeCodeFloat = 2;

// DLPack packs `lanes` elements into each vector element; 1 is a plain scalar.
// The postprocessing kernels index the buffer as scalars, so anything else is
// a different memory layout and is rejected rather than misread.
constexpr uint16_t kDLDataTypeLanesScalar = 1;

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

// A DLPack tensor may carry explicit strides; empty means "contiguous, infer
// row-major from shape". The postprocessing kernels index the tensor as a dense
// row-major buffer, so any other layout must be rejected rather than silently
// misread.
bool IsContiguousRowMajor(const tensor_msgs::msg::ExperimentalTensor & tensor)
{
  if (tensor.strides.empty()) {return true;}
  if (tensor.strides.size() != tensor.shape.size()) {return false;}
  int64_t expected = 1;
  for (size_t i = tensor.shape.size(); i > 0; --i) {
    if (tensor.strides[i - 1] != expected) {return false;}
    expected *= tensor.shape[i - 1];
  }
  return true;
}

Shape ExtractShape(
  const tensor_msgs::msg::ExperimentalTensor & tensor,
  DataFormat data_format)
{
  Shape shape{};
  const auto & dims = tensor.shape;
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
            "Tensor has insufficient dimensions for DataFormat::" +
            std::string(format_name) + ": expected at least " +
            std::to_string(required_dims) + ", got " + std::to_string(dims.size()));
  }
  switch (data_format) {
    case DataFormat::kHWC:
      shape.height = static_cast<int32_t>(dims[0]);
      shape.width = static_cast<int32_t>(dims[1]);
      shape.channels = static_cast<int32_t>(dims[2]);
      break;
    case DataFormat::kNCHW:
      shape.channels = static_cast<int32_t>(dims[1]);
      shape.height = static_cast<int32_t>(dims[2]);
      shape.width = static_cast<int32_t>(dims[3]);
      break;
    case DataFormat::kNHWC:
      shape.height = static_cast<int32_t>(dims[1]);
      shape.width = static_cast<int32_t>(dims[2]);
      shape.channels = static_cast<int32_t>(dims[3]);
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
  data_format_(declare_parameter<std::string>("data_format", "NHWC"))
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
  // Accept GPU-backed (cuda) buffers while staying compatible with CPU-backed
  // publishers; from_input_buffer promotes CPU buffers as needed.
  sub_options.acceptable_buffer_backends = "any";
  tensor_sub_ = create_subscription<isaac_ros_tensor_msgs::msg::TensorList>(
    "tensor_sub", input_qos,
    std::bind(&UNetDecoderNode::TensorCallback, this, std::placeholders::_1),
    sub_options);

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  raw_mask_pub_ = create_publisher<sensor_msgs::msg::Image>(
    "unet/raw_segmentation_mask", output_qos, pub_options);
  colored_mask_pub_ = create_publisher<sensor_msgs::msg::Image>(
    "unet/colored_segmentation_mask", output_qos, pub_options);
}

void UNetDecoderNode::TensorCallback(
  const isaac_ros_tensor_msgs::msg::TensorList::ConstSharedPtr & msg)
{
  if (msg->tensors.empty()) {
    RCLCPP_ERROR(get_logger(), "Received empty tensor list!");
    return;
  }

  const auto & tensor = msg->tensors[0];
  if (tensor.dtype_lanes != kDLDataTypeLanesScalar) {
    RCLCPP_ERROR(
      get_logger(),
      "Received vectorized tensor (dtype_lanes = %u), only scalar tensors are supported.",
      static_cast<unsigned int>(tensor.dtype_lanes));
    return;
  }
  if (!IsContiguousRowMajor(tensor)) {
    RCLCPP_ERROR(
      get_logger(),
      "Received tensor with non-contiguous strides, only row-major layout is supported.");
    return;
  }

  Shape shape = ExtractShape(tensor, data_format_value_);

  if (shape.channels > kMaxChannelCount) {
    RCLCPP_ERROR(
      get_logger(), "Received %d channels, max is %ld",
      shape.channels, kMaxChannelCount);
    return;
  }

  // Allocate raw mask output (mono8) with a CUDA-backed buffer
  auto raw_mask_msg = std::make_unique<sensor_msgs::msg::Image>();
  raw_mask_msg->header = msg->header;
  raw_mask_msg->height = static_cast<uint32_t>(shape.height);
  raw_mask_msg->width = static_cast<uint32_t>(shape.width);
  raw_mask_msg->encoding = sensor_msgs::image_encodings::MONO8;
  raw_mask_msg->is_bigendian = 0;
  size_t raw_step = static_cast<size_t>(shape.width);
  raw_mask_msg->step = static_cast<uint32_t>(raw_step);
  raw_mask_msg->data = cuda_buffer_backend::allocate_buffer(
    raw_step * static_cast<size_t>(shape.height));

  // Allocate colored mask output (rgb8 or bgr8) with a CUDA-backed buffer
  auto colored_mask_msg = std::make_unique<sensor_msgs::msg::Image>();
  colored_mask_msg->header = msg->header;
  colored_mask_msg->height = static_cast<uint32_t>(shape.height);
  colored_mask_msg->width = static_cast<uint32_t>(shape.width);
  colored_mask_msg->encoding = color_segmentation_mask_encoding_;
  colored_mask_msg->is_bigendian = 0;
  size_t color_step = static_cast<size_t>(shape.width) * 3;
  colored_mask_msg->step = static_cast<uint32_t>(color_step);
  colored_mask_msg->data = cuda_buffer_backend::allocate_buffer(
    color_step * static_cast<size_t>(shape.height));

  {
    // The Read/WriteHandles below are the RAII owners of this scope's GPU access:
    // each gates on the producer's write event (when the buffer carries one),
    // keeps the CudaBuffer storage alive, and settles its bookkeeping on
    // destruction. They must therefore outlive all GPU work below and be
    // destroyed before the messages are published.
    // The uint8_t pointers taken from them are non-owning device views into
    // buffer-owned memory -- they must not be wrapped in an owning smart pointer,
    // which would free storage this scope does not own.
    auto tensor_read_handle = cuda_buffer_backend::from_input_buffer(tensor.data, *cuda_stream_);
    // byte_offset is nonzero when the message carries a view into a larger
    // allocation; the tensor's first element lives at data + byte_offset.
    const uint8_t * tensor_gpu_ptr = tensor_read_handle.get_ptr() + tensor.byte_offset;

    auto raw_write_handle = cuda_buffer_backend::from_output_buffer(
      raw_mask_msg->data, *cuda_stream_);
    uint8_t * raw_mask_ptr = raw_write_handle.get_ptr();

    // Run postprocessing kernel
    if (network_output_type_value_ == NetworkOutputType::kArgmax) {
      if (tensor.dtype_code == kDLDataTypeCodeInt && tensor.dtype_bits == 32) {
        CopyTensorData<int32_t>(
          network_output_type_value_, data_format_value_, shape,
          reinterpret_cast<const int32_t *>(tensor_gpu_ptr),
          raw_mask_ptr, *cuda_stream_);
      } else if (tensor.dtype_code == kDLDataTypeCodeInt && tensor.dtype_bits == 64) {
        CopyTensorData<int64_t>(
          network_output_type_value_, data_format_value_, shape,
          reinterpret_cast<const int64_t *>(tensor_gpu_ptr),
          raw_mask_ptr, *cuda_stream_);
      } else {
        RCLCPP_ERROR(
          get_logger(),
          "Unsupported tensor element type for argmax: expected int32 or int64, got "
          "{dtype_code = %u, dtype_bits = %u}.",
          static_cast<unsigned int>(tensor.dtype_code),
          static_cast<unsigned int>(tensor.dtype_bits));
        return;
      }
    } else {
      if (tensor.dtype_code != kDLDataTypeCodeFloat || tensor.dtype_bits != 32) {
        RCLCPP_ERROR(
          get_logger(),
          "Unsupported tensor element type for %s: expected float32, got "
          "{dtype_code = %u, dtype_bits = %u}.",
          network_output_type_.c_str(),
          static_cast<unsigned int>(tensor.dtype_code),
          static_cast<unsigned int>(tensor.dtype_bits));
        return;
      }
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

    // Non-owning device view, owned by color_write_handle (see above).
    auto color_write_handle = cuda_buffer_backend::from_output_buffer(
      colored_mask_msg->data, *cuda_stream_);
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

    // Sync before publishing. The handles do NOT order this for us: a WriteHandle
    // records its completion event only if the buffer already owns one, and
    // CudaBuffer::write_event_ is set exclusively when the backend imports an IPC
    // handle on the *subscriber* side. Buffers minted by allocate_buffer() have no
    // event, so consumers get a zeroed ipc_event_handle and never wait on this
    // stream -- the masks must be complete before they go out.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_), "Failed to sync stream");
  }  // handles destroyed here → read events recorded, write access finalized

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
