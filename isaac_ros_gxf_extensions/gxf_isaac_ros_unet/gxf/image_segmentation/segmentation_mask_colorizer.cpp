// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "segmentation_mask_colorizer.hpp"

#include <string>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/timestamp.hpp"
#include "segmentation_mask_colorizer.cu.hpp"
#include "segmentation_postprocessing_utils.hpp"

namespace nvidia
{
namespace isaac_ros
{

namespace
{

gxf::Expected<void> AddColorToSegmentationMask(
  gxf::Handle<gxf::VideoBuffer> output_colored_video_buffer,
  gxf::Handle<gxf::VideoBuffer> input_raw_segmentation_mask, ColorImageEncodings encoding,
  const ArrayView<int64_t> & color_palette, cudaStream_t stream)
{
  uint8_t * output_buffer = static_cast<uint8_t *>(output_colored_video_buffer->pointer());
  if (!output_buffer) {
    GXF_LOG_ERROR("Error could not get colorized video buffer data!");
    return gxf::Unexpected{GXF_FAILURE};
  }

  uint8_t * input_buffer = static_cast<uint8_t *>(input_raw_segmentation_mask->pointer());
  if (!input_buffer) {
    GXF_LOG_ERROR("Error could not get raw segmentation video buffer data!");
    return gxf::Unexpected{GXF_FAILURE};
  }

  ColorizeSegmentationMask(
    output_buffer, input_raw_segmentation_mask->video_frame_info().width,
    input_raw_segmentation_mask->video_frame_info().height, encoding,
    input_buffer, color_palette, stream);
  cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess) {
    GXF_LOG_ERROR("Error while executing kernel: %s", cudaGetErrorString(result));
    return gxf::Unexpected{GXF_FAILURE};
  }
  return gxf::Success;
}

gxf::Expected<void> AddInputTimestampToOutput(gxf::Entity & output, gxf::Entity input)
{
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

}  // namespace

gxf_result_t SegmentationMaskColorizer::registerInterface(gxf::Registrar * registrar)
{
  gxf::Expected<void> result;
  result &= registrar->parameter(
    input_, "raw_segmentation_mask_input", "Raw Segmentation Mask Input",
    "The raw segmentation mask input. "
    "This is expected to be a video buffer that will be forwarded");
  result &= registrar->parameter(
    colored_segmentation_output_, "colored_segmentation_mask_output",
    "Colored Segmentation Mask Output",
    "The colorized segmentation mask output. "
    "This is expected to be a video buffer that that will be filled using the color palette");
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "The allocator");
  result &= registrar->parameter(
    color_palette_vec_, "color_palette", "Color Palette",
    "A vector of integers, where each element represents the RGB hex code for "
    "the corresponding class label. Note: only the first 24 bits are used");
  result &= registrar->parameter(
    color_segmentation_mask_encoding_str_, "color_segmentation_mask_encoding",
    "Color Segmentation Mask Encoding",
    "The encoding of the colored segmentation mask. This should be either rgb8 or bgr8");
  return gxf::ToResultCode(result);
}

gxf_result_t SegmentationMaskColorizer::start()
{
  if (color_palette_vec_.get().empty()) {
    GXF_LOG_ERROR("Error: received empty color palette!");
    return GXF_FAILURE;
  }

  int64_t * data{nullptr};
  cudaError_t result =
    cudaMalloc(&data, sizeof(int64_t) * color_palette_vec_.get().size());
  if (result != cudaSuccess) {return GXF_FAILURE;}
  color_palette_.data.reset(data);

  result = cudaMemcpy(
    color_palette_.data.get(), color_palette_vec_.get().data(),
    color_palette_vec_.get().size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {return GXF_FAILURE;}

  if (color_segmentation_mask_encoding_str_.get() == std::string{"rgb8"}) {
    color_segmentation_mask_encoding_ = ColorImageEncodings::kRGB8;
  } else if (color_segmentation_mask_encoding_str_.get() == std::string{"bgr8"}) {
    color_segmentation_mask_encoding_ = ColorImageEncodings::kBGR8;
  } else {
    GXF_LOG_ERROR(
      "Received unsupport color encoding: %s",
      color_segmentation_mask_encoding_str_.get().c_str());
    return GXF_FAILURE;
  }

  color_palette_.size = color_palette_vec_.get().size();
  return GXF_SUCCESS;
}

gxf_result_t SegmentationMaskColorizer::tick()
{
  gxf::Entity input_entity;
  gxf::Entity output_entity;
  gxf::Handle<gxf::VideoBuffer> raw_segmentation_mask;
  std::string timestamp_name{"timestamp"};
  gxf::Handle<gxf::Timestamp> input_timestamp;
  gxf::Handle<gxf::CudaStream> stream;
  return gxf::ToResultCode(
    input_->receive()
    .assign_to(input_entity)
    .and_then([&]() {return gxf::Entity::New(context());})
    .assign_to(output_entity)
    .and_then([&]() {return input_entity.get<gxf::CudaStreamId>();})
    .map(
      [&](gxf::Handle<gxf::CudaStreamId> cuda_stream_id) {
        return gxf::Handle<gxf::CudaStream>::Create(
          cuda_stream_id.context(),
          cuda_stream_id->stream_cid);
      })
    .assign_to(stream)
    .and_then([&]() {return stream->stream();})
    .and_then(
      [&]() {
        return generateColorizedSegmentationMask(output_entity, input_entity, stream);
      })
    .and_then([&]() {return AddInputTimestampToOutput(output_entity, input_entity);})
    .and_then(
      [&]() {
        return cudaStreamSynchronize(stream->stream().value()) == cudaSuccess ?
        gxf::Success :
        gxf::Unexpected{GXF_FAILURE};
      })
    .and_then([&]() {return colored_segmentation_output_->publish(output_entity);}));
}

gxf::Expected<void> SegmentationMaskColorizer::generateColorizedSegmentationMask(
  gxf::Entity & output, gxf::Entity input, gxf::Handle<gxf::CudaStream> stream)
{
  gxf::Handle<gxf::VideoBuffer> raw_segmentation_mask;
  gxf::Handle<gxf::VideoBuffer> colored_segmentation_mask;
  return input.get<gxf::VideoBuffer>()
         .assign_to(raw_segmentation_mask)
         .and_then([&]() {return output.add<gxf::VideoBuffer>();})
         .assign_to(colored_segmentation_mask)
         .and_then(
    [&]() {
      return createColorSegmentationMask(colored_segmentation_mask, raw_segmentation_mask);
    })
         .and_then(
    [&]() {
      return AddColorToSegmentationMask(
        colored_segmentation_mask, raw_segmentation_mask,
        color_segmentation_mask_encoding_, color_palette_,
        stream->stream().value());
    });
}

gxf::Expected<void> SegmentationMaskColorizer::createColorSegmentationMask(
  gxf::Handle<gxf::VideoBuffer> colored_segmentation_mask,
  gxf::Handle<gxf::VideoBuffer> raw_segmentation_mask)
{
  gxf::Expected<void> maybe_allocation = gxf::Unexpected{GXF_FAILURE};
  switch (color_segmentation_mask_encoding_) {
    case ColorImageEncodings::kRGB8:
      maybe_allocation = AllocateVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB>(
        colored_segmentation_mask, raw_segmentation_mask->video_frame_info().width,
        raw_segmentation_mask->video_frame_info().height, gxf::MemoryStorageType::kDevice,
        allocator_.get());
      break;
    case ColorImageEncodings::kBGR8:
      maybe_allocation = AllocateVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR>(
        colored_segmentation_mask, raw_segmentation_mask->video_frame_info().width,
        raw_segmentation_mask->video_frame_info().height, gxf::MemoryStorageType::kDevice,
        allocator_.get());
      break;
    default: {
        GXF_LOG_ERROR("Error: received unexpected encoding!");
        break;
      }
  }

  if (!maybe_allocation) {
    GXF_LOG_ERROR("Error allocating colorized video buffer!");
    return gxf::ForwardError(maybe_allocation);
  }

  return gxf::Success;
}

gxf_result_t SegmentationMaskColorizer::stop()
{
  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
