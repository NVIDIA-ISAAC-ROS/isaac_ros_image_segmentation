// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "segmentation_postprocessor.hpp"

#include <string>
#include <utility>

#include "cuda.h"
#include "cuda_runtime.h"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"
#include "segmentation_postprocessing_utils.hpp"
#include "segmentation_postprocessor.cu.hpp"

namespace nvidia {
namespace isaac_ros {

namespace {

gxf::Expected<gxf::Handle<gxf::Tensor>> GetTensor(gxf::Entity entity, const char* tensor_name) {
  // Try getting the named tensor first
  auto maybe_tensor = entity.get<gxf::Tensor>(tensor_name);

  // Default to getting any tensor
  if (!maybe_tensor) { maybe_tensor = entity.get<gxf::Tensor>(); }

  if (!maybe_tensor) { GXF_LOG_ERROR("Failed to get any tensor from input message!"); }

  return maybe_tensor;
}

gxf::Expected<Shape> GetShape(gxf::Handle<gxf::Tensor> in_tensor, DataFormat data_format) {
  Shape shape{};
  switch (data_format) {
    case DataFormat::kHWC: {
      shape.height = in_tensor->shape().dimension(0);
      shape.width = in_tensor->shape().dimension(1);
      shape.channels = in_tensor->shape().dimension(2);
    } break;
    case DataFormat::kNCHW: {
      shape.channels = in_tensor->shape().dimension(1);
      shape.height = in_tensor->shape().dimension(2);
      shape.width = in_tensor->shape().dimension(3);
    } break;
    case DataFormat::kNHWC: {
      shape.height = in_tensor->shape().dimension(1);
      shape.width = in_tensor->shape().dimension(2);
      shape.channels = in_tensor->shape().dimension(3);
    } break;
  }

  if (shape.channels > kMaxChannelCount) {
    GXF_LOG_ERROR("Received %d input channels, which is larger than the maximum allowable %d",
                  shape.channels, kMaxChannelCount);
    return gxf::Unexpected{GXF_FAILURE};
  }

  return shape;
}

gxf::Expected<void> ConvertTensorToVideoBuffer(NetworkOutputType network_output_type,
                                               DataFormat data_format, Shape shape,
                                               gxf::Handle<gxf::Tensor> in_tensor,
                                               gxf::Handle<gxf::VideoBuffer> output_video_buffer,
                                               cudaStream_t stream) {
  gxf::Expected<const int32_t*> in_tensor_data = in_tensor->data<int32_t>();
  if (!in_tensor_data) {
    GXF_LOG_ERROR("Failed to get input tensor data!");
    return gxf::ForwardError(in_tensor_data);
  }

  uint8_t* output_video_buffer_data = static_cast<uint8_t*>(output_video_buffer->pointer());
  if (!output_video_buffer_data) {
    GXF_LOG_ERROR("Failed to get video buffer data!");
    return gxf::Unexpected{GXF_FAILURE};
  }

  CopyTensorData(network_output_type, data_format, shape, in_tensor_data.value(),
                 output_video_buffer_data, stream);
  cudaError_t kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    GXF_LOG_ERROR("Error while executing kernel: %s", cudaGetErrorString(kernel_result));
    return gxf::Unexpected{GXF_FAILURE};
  }

  return gxf::Success;
}

gxf::Expected<void> GenerateSegmentationMask(NetworkOutputType network_output_type,
                                             DataFormat data_format, Shape shape,
                                             gxf::Handle<gxf::Tensor> in_tensor,
                                             gxf::Handle<gxf::VideoBuffer> output_video_buffer,
                                             cudaStream_t stream) {
  gxf::Expected<const float*> in_tensor_data = in_tensor->data<float>();
  if (!in_tensor_data) {
    GXF_LOG_ERROR("Failed to get input tensor data!");
    return gxf::ForwardError(in_tensor_data);
  }

  uint8_t* output_video_buffer_data = static_cast<uint8_t*>(output_video_buffer->pointer());
  if (!output_video_buffer_data) {
    GXF_LOG_ERROR("Failed to get video buffer data!");
    return gxf::Unexpected{GXF_FAILURE};
  }

  cuda_postprocess(network_output_type, data_format, shape, in_tensor_data.value(),
                   output_video_buffer_data, stream);
  cudaError_t kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    GXF_LOG_ERROR("Error while executing kernel: %s", cudaGetErrorString(kernel_result));
    return gxf::Unexpected{GXF_FAILURE};
  }

  return gxf::Success;
}

gxf::Expected<void> AddInputTimestampToOutput(gxf::Entity& output, gxf::Entity input) {
  std::string timestamp_name{"timestamp"};
  auto maybe_timestamp = input.get<gxf::Timestamp>(timestamp_name.c_str());

  // Default to unnamed
  if (!maybe_timestamp) {
    timestamp_name = std::string{""};
    maybe_timestamp = input.get<gxf::Timestamp>(timestamp_name.c_str());
  }

  if (!maybe_timestamp) {
    GXF_LOG_ERROR("Failed to get input timestamp!");
    return gxf::ForwardError(maybe_timestamp);
  }

  auto maybe_out_timestamp = output.add<gxf::Timestamp>(timestamp_name.c_str());
  if (!maybe_out_timestamp) {
    GXF_LOG_ERROR("Failed to add timestamp to output message!");
    return gxf::ForwardError(maybe_out_timestamp);
  }

  *maybe_out_timestamp.value() = *maybe_timestamp.value();
  return gxf::Success;
}

gxf::Expected<void> AddStreamToOutput(gxf::Entity& output, gxf::Handle<gxf::CudaStream> stream) {
  auto maybe_stream_id = output.add<gxf::CudaStreamId>();
  if (!maybe_stream_id) { return gxf::ForwardError(maybe_stream_id); }
  maybe_stream_id.value()->stream_cid = stream.cid();
  if (maybe_stream_id.value()->stream_cid == kNullUid) {
    GXF_LOG_ERROR("Error: cuda stream handle is null!");
    return gxf::Unexpected{GXF_FAILURE};
  }
  return gxf::Success;
}

}  // namespace

gxf_result_t SegmentationPostprocessor::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(in_, "in", "Input", "Input channel.");
  result &= registrar->parameter(in_tensor_name_, "in_tensor_name", "InputTensorName",
                                 "Name of the input tensor.", std::string(""));
  result &= registrar->parameter(network_output_type_, "network_output_type", "NetworkOutputType",
                                 "Network output type.", std::string("softmax"));
  result &= registrar->parameter(out_, "out", "Output", "Output channel.");
  result &= registrar->parameter(data_format_, "data_format", "DataFormat",
                                 "Data format of network output", std::string("NHWC"));
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "Output Allocator");
  result &= registrar->parameter(cuda_stream_pool_, "cuda_stream_pool");
  return gxf::ToResultCode(result);
}

gxf_result_t SegmentationPostprocessor::start() {
  const std::string& network_output_type_name = network_output_type_.get();
  if (network_output_type_name == "sigmoid") {
    network_output_type_value_ = NetworkOutputType::kSigmoid;
  } else if (network_output_type_name == "softmax") {
    network_output_type_value_ = NetworkOutputType::kSoftmax;
  } else if (network_output_type_name == "argmax") {
    network_output_type_value_ = NetworkOutputType::kArgmax;
  } else {
    GXF_LOG_ERROR("Unsupported network type %s", network_output_type_name.c_str());
    return GXF_FAILURE;
  }
  const std::string& data_format_name = data_format_.get();
  if (data_format_name == "NCHW") {
    data_format_value_ = DataFormat::kNCHW;
  } else if (data_format_name == "HWC") {
    data_format_value_ = DataFormat::kHWC;
  } else if (data_format_name == "NHWC") {
    data_format_value_ = DataFormat::kNHWC;
  } else {
    GXF_LOG_ERROR("Unsupported format type %s", data_format_name.c_str());
    return GXF_FAILURE;
  }

  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) { return gxf::ToResultCode(maybe_stream); }

  stream_ = std::move(maybe_stream.value());
  if (!stream_->stream()) {
    GXF_LOG_ERROR("Error: allocated stream is not initialized!");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t SegmentationPostprocessor::tick() {
  // Process input message
  const auto in_message = in_->receive();

  if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }

  auto maybe_tensor = GetTensor(in_message.value(), in_tensor_name_.get().c_str());
  if (!maybe_tensor) { return gxf::ToResultCode(maybe_tensor); }

  gxf::Handle<gxf::Tensor> in_tensor = maybe_tensor.value();

  auto maybe_shape = GetShape(in_tensor, data_format_value_);
  if (!maybe_shape) { return gxf::ToResultCode(maybe_shape); }

  Shape shape = maybe_shape.value();

  auto out_message = gxf::Entity::New(context());
  if (!out_message) {
    GXF_LOG_ERROR("Failed to allocate message");
    return gxf::ToResultCode(out_message);
  }

  auto out_video_buffer = out_message.value().add<gxf::VideoBuffer>();
  if (!out_video_buffer) {
    GXF_LOG_ERROR("Failed to allocate output video buffer");
    return gxf::ToResultCode(out_video_buffer);
  }

  // Allocate and convert output buffer on the device.
  auto maybe_allocation = AllocateVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY>(
      out_video_buffer.value(), shape.width, shape.height, gxf::MemoryStorageType::kDevice,
      allocator_.get());
  if (!maybe_allocation) {
    GXF_LOG_ERROR("Failed to allocate output video buffer's memory.");
    return gxf::ToResultCode(maybe_allocation);
  }

  // Update the CUDA device to run the kernel on
  if (stream_->dev_id() >= 0) {
    if (cudaSetDevice(stream_->dev_id()) != cudaSuccess) {
      GXF_LOG_ERROR("Failed to set device_id: %d", stream_->dev_id());
      return GXF_FAILURE;
    }
  }
  auto maybe_cuda_stream = stream_->stream();
  if (!maybe_cuda_stream) { return gxf::ToResultCode(maybe_cuda_stream); }

  cudaStream_t cuda_stream = maybe_cuda_stream.value();

  gxf::Expected<void> maybe_kernel_result;
  if (network_output_type_value_ == NetworkOutputType::kArgmax) {
    maybe_kernel_result =
        ConvertTensorToVideoBuffer(network_output_type_value_, data_format_value_, shape, in_tensor,
                                   out_video_buffer.value(), cuda_stream);
  } else {
    maybe_kernel_result =
        GenerateSegmentationMask(network_output_type_value_, data_format_value_, shape, in_tensor,
                                 out_video_buffer.value(), cuda_stream);
  }

  auto maybe_added_timestamp = AddInputTimestampToOutput(out_message.value(), in_message.value());
  if (!maybe_added_timestamp) { return gxf::ToResultCode(maybe_added_timestamp); }

  auto maybe_added_stream = AddStreamToOutput(out_message.value(), stream_);
  if (!maybe_added_stream) { return gxf::ToResultCode(maybe_added_stream); }

  return gxf::ToResultCode(out_->publish(std::move(out_message.value())));
}

gxf_result_t SegmentationPostprocessor::stop() {
  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
