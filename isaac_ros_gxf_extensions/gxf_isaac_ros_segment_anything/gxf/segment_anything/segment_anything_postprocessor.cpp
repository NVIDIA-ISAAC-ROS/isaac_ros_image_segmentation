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
#include "segment_anything_postprocessor.hpp"

#include <string>
#include <utility>

#include "cuda.h"
#include "cuda_runtime.h"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"
#include "segment_anything_postprocessor.cu.hpp"

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

gxf::Expected<Shape> GetShape(gxf::Handle<gxf::Tensor> in_tensor) {
  Shape shape{};

  shape.batch_size = in_tensor->shape().dimension(0);
  shape.channels = in_tensor->shape().dimension(1);
  shape.height = in_tensor->shape().dimension(2);
  shape.width = in_tensor->shape().dimension(3);

  // Only single channel
  if (shape.channels != kExpectedChannelCount) {
    GXF_LOG_ERROR("Received %d input channels, which is larger than the maximum allowable %d",
                  shape.channels, kExpectedChannelCount);
    return gxf::Unexpected{GXF_FAILURE};
  }

  return shape;
}

gxf::Expected<void> GenerateSegmentationMask(Shape shape,
                                             gxf::Handle<gxf::Tensor> in_tensor,
                                             uint8_t* output_raw_buffer,
                                             cudaStream_t stream) {
  gxf::Expected<const float*> in_tensor_data = in_tensor->data<float>();
  if (!in_tensor_data) {
    GXF_LOG_ERROR("Failed to get input tensor data!");
    return gxf::ForwardError(in_tensor_data);
  }

  cuda_postprocess(shape, in_tensor_data.value(), output_raw_buffer,stream);
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

gxf_result_t SegmentAnythingPostprocessor::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(in_, "in", "Input", "Input channel.");
  result &= registrar->parameter(in_tensor_name_, "in_tensor_name", "InputTensorName",
                                 "Name of the input tensor.", std::string(""));
  result &= registrar->parameter(out_, "out", "Output", "Output channel.");
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "Output Allocator");
  result &= registrar->parameter(cuda_stream_pool_, "cuda_stream_pool");
  return gxf::ToResultCode(result);
}

gxf_result_t SegmentAnythingPostprocessor::start() {
  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) { return gxf::ToResultCode(maybe_stream); }

  stream_ = std::move(maybe_stream.value());
  if (!stream_->stream()) {
    GXF_LOG_ERROR("Error: allocated stream is not initialized!");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t SegmentAnythingPostprocessor::tick() {
  // Process input message
  const auto in_message = in_->receive();

  if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }

  auto maybe_tensor = GetTensor(in_message.value(), in_tensor_name_.get().c_str());
  if (!maybe_tensor) { return gxf::ToResultCode(maybe_tensor); }

  gxf::Handle<gxf::Tensor> in_tensor = maybe_tensor.value();

  auto maybe_shape = GetShape(in_tensor);
  if (!maybe_shape) { return gxf::ToResultCode(maybe_shape); }

  Shape shape = maybe_shape.value();

  auto out_message = gxf::Entity::New(context());

  if (!out_message) {
    GXF_LOG_ERROR("Failed to allocate message");
    return gxf::ToResultCode(out_message);
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

  auto raw_tensor = out_message.value().add<gxf::Tensor>();

  if (!raw_tensor) {
    GXF_LOG_ERROR("Failed to allocate output raw tensor");
    return gxf::ToResultCode(raw_tensor);
  }

  auto result = raw_tensor.value()->reshapeCustom(
  in_tensor->shape(), gxf::PrimitiveType::kUnsigned8,
      gxf::PrimitiveTypeSize(gxf::PrimitiveType::kUnsigned8),
  gxf::Unexpected{GXF_UNINITIALIZED_VALUE}, in_tensor->storage_type(), allocator_.get());

  uint8_t* output_video_buffer_data = static_cast<uint8_t*>(raw_tensor.value()->pointer());
  if (!output_video_buffer_data) {
    GXF_LOG_ERROR("Failed to get raw tensor buffer data!");
    return GXF_FAILURE;
  }

  maybe_kernel_result =
      GenerateSegmentationMask(shape, in_tensor, output_video_buffer_data, cuda_stream);

  auto maybe_added_timestamp = AddInputTimestampToOutput(out_message.value(), in_message.value());
  if (!maybe_added_timestamp) { return gxf::ToResultCode(maybe_added_timestamp); }


  auto maybe_added_stream = AddStreamToOutput(out_message.value(), stream_);
  if (!maybe_added_stream) { return gxf::ToResultCode(maybe_added_stream); }

  // Sync the stream, before sending it across the framework.
  // This is important to do since Isaac does not support non synced work across
  // codelets and multiple nodes.
  cudaError_t sync_result = cudaStreamSynchronize(cuda_stream);
  if (sync_result != cudaSuccess) {
    GXF_LOG_ERROR("Error while synchronizing CUDA stream: %s", cudaGetErrorString(sync_result));
    return GXF_FAILURE;
  }

  return gxf::ToResultCode(out_->publish(std::move(out_message.value())));
}

gxf_result_t SegmentAnythingPostprocessor::stop() {
  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
