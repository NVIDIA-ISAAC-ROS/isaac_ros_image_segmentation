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
#include "segment_anything_prompt_processor.hpp"

#include <string>
#include <utility>

#include "cuda.h"
#include "cuda_runtime.h"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"
#include "detection2_d_array_message.hpp"

namespace nvidia {
namespace isaac_ros {

namespace {

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
}  // namespace

gxf_result_t SegmentAnythingPromptProcessor::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(in_, "in", "Input", "Input channel.");
  result &= registrar->parameter(out_points_, "out_points", "Output", "Output channel.");
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "Output Allocator");
  result &= registrar->parameter(orig_img_dim_, "orig_img_dim");
  result &= registrar->parameter(max_batch_size_, "max_batch_size");
  result &= registrar->parameter(prompt_type_name_, "prompt_type_name");
  result &= registrar->parameter(has_input_mask_, "has_input_mask");
  return gxf::ToResultCode(result);
}

gxf_result_t SegmentAnythingPromptProcessor::start() {

  uint32_t orig_width = orig_img_dim_.get()[1];
  uint32_t orig_height = orig_img_dim_.get()[0];

  if (orig_width > orig_height) {
    resized_width_ = IMAGE_WIDTH_;
    resized_height_ = int((float(resized_width_) / orig_width) * orig_height);
  } else { 
    resized_height_ = IMAGE_HEIGHT_;
    resized_width_ = int((float(resized_height_) / orig_height) * orig_width);
  }

  prompt_type_value_ = prompt_type_name_.get() == "bbox" ? PromptType::kBbox : PromptType::kPoint;

  return GXF_SUCCESS;
}

gxf_result_t SegmentAnythingPromptProcessor::tick() {
  // Process input message
  const auto in_message = in_->receive();
  auto out_message = gxf::Entity::New(context());

  if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }
  Detection2DParts parts;
  auto detection2_d_parts_expected = nvidia::isaac_ros::GetDetection2DList(
    in_message.value());
  
  if (!detection2_d_parts_expected) { return gxf::ToResultCode(detection2_d_parts_expected); }
  
  auto detection2_d_parts = detection2_d_parts_expected.value();

  // Extract detection2_d array to a struct type defined in detection2_d.hpp
  std::vector<nvidia::isaac_ros::Detection2D> detections =
    *(detection2_d_parts.detection2_d_array);
  
  auto maybe_added_timestamp = AddInputTimestampToOutput(out_message.value(), in_message.value());
  if (!maybe_added_timestamp) {
    GXF_LOG_ERROR("Failed to add timestamp to output msg");
    return gxf::ToResultCode(maybe_added_timestamp);
  }
  
  // Add invalid_frame tensor and publish it.
  // invalid_frame tensor is used by msg compositor to disqualify this input.
  if(detections.size() == 0){
    GXF_LOG_INFO("No input prompt found. No inference would run on this frame.");
    auto invalid_frame = out_message.value().add<gxf::Tensor>("invalid_frame");
    if (!invalid_frame) {
      GXF_LOG_ERROR("Failed to allocate invalid_frame tensor");
      return gxf::ToResultCode(invalid_frame);
    }
    out_points_->publish(std::move(out_message.value()));
  return GXF_SUCCESS;
  }

  uint32_t batch_size = 1;
  uint32_t num_points = num_points_;
  if (prompt_type_value_ == PromptType::kBbox) { 
    batch_size = max_batch_size_ < detections.size() ? max_batch_size_ : detections.size();
  }

  if (prompt_type_value_ == PromptType::kPoint){
    // +1 for padding point
    num_points = detections.size() + 1;
  }

  std::vector<float> prompt_vec;
  std::vector<float> label_vec;

  detectionToSAMPrompt(detections, prompt_vec, label_vec);
  
  if (!out_message) {
    GXF_LOG_ERROR("Failed to allocate message");
    return gxf::ToResultCode(out_message);
  }

  auto bbox_tensor =  out_message.value().add<gxf::Tensor>("points");
  if (!bbox_tensor) {
    GXF_LOG_ERROR("Failed to allocate bbox tensor");
    return gxf::ToResultCode(bbox_tensor);
  }

  auto labels_tensor = out_message.value().add<gxf::Tensor>("labels");
  if (!labels_tensor) {
    GXF_LOG_ERROR("Failed to allocate label tensor");
    return gxf::ToResultCode(labels_tensor);
  }

  auto img_size_tensor = out_message.value().add<gxf::Tensor>("orig_img_dims");
  if (!img_size_tensor) {
    GXF_LOG_ERROR("Failed to allocate img_size_tensor tensor");
    return gxf::ToResultCode(img_size_tensor);
  }

  auto has_input_mask_tensor = out_message.value().add<gxf::Tensor>("has_input_mask");
  if (!has_input_mask_tensor) {
    GXF_LOG_ERROR("Failed to allocate has_input_mask_tensor tensor");
    return gxf::ToResultCode(has_input_mask_tensor);
  }

  // Create Tensor for prompt data.
  gxf::Shape prompt_tensor_shape({batch_size,num_points,2});
  uint32_t buffer_size{batch_size*num_points*2*sizeof(float)};
  auto result = bbox_tensor.value()->reshapeCustom(
  prompt_tensor_shape, gxf::PrimitiveType::kFloat32,
      gxf::PrimitiveTypeSize(gxf::PrimitiveType::kFloat32),
  gxf::Unexpected{GXF_UNINITIALIZED_VALUE}, gxf::MemoryStorageType::kDevice, allocator_.get());  
  if(!result){
    GXF_LOG_ERROR("Error allocating tensor for points tensor.");
    return gxf::ToResultCode(result);
  }
  cudaMemcpy(bbox_tensor.value()->pointer(), prompt_vec.data(), buffer_size,cudaMemcpyHostToDevice);
  
  // Create Tensor for labels corresponding to each point
  gxf::Shape label_tensor_shape({batch_size,num_points});
  uint32_t buffer_size_label{batch_size*num_points*sizeof(float)};
  auto label_tensor_result = labels_tensor.value()->reshapeCustom(
  label_tensor_shape, gxf::PrimitiveType::kFloat32,
      gxf::PrimitiveTypeSize(gxf::PrimitiveType::kFloat32),
  gxf::Unexpected{GXF_UNINITIALIZED_VALUE}, gxf::MemoryStorageType::kDevice, allocator_.get());
  if(!label_tensor_result){
    GXF_LOG_ERROR("Error allocating tensor for label tensor.");
    return gxf::ToResultCode(label_tensor_result);
  }
  cudaMemcpy(labels_tensor.value()->pointer(), label_vec.data(), buffer_size_label,cudaMemcpyHostToDevice); 

  // Create Tensor to put has_input_mask data
  gxf::Shape has_in_mask({1});
  std::vector<float> has_input_mask(1);
  has_input_mask[0] = has_input_mask_.get() ? 1.0 : 0.0;
  auto has_input_mask_result = has_input_mask_tensor.value()->reshapeCustom(
  has_in_mask, gxf::PrimitiveType::kFloat32,
      gxf::PrimitiveTypeSize(gxf::PrimitiveType::kFloat32),
  gxf::Unexpected{GXF_UNINITIALIZED_VALUE}, gxf::MemoryStorageType::kDevice, allocator_.get());
  if(!has_input_mask_result){
    GXF_LOG_ERROR("Error allocating tensor for has input mask tensor.");
    return gxf::ToResultCode(has_input_mask_result);
  }
  cudaMemcpy(has_input_mask_tensor.value()->pointer(), has_input_mask.data(), sizeof(float), cudaMemcpyHostToDevice);
  
  // Create Tensor to put original image shape data
  gxf::Shape img_size_tensor_shape({2});
  std::vector<float> img_size(2);
  img_size[0] = orig_img_dim_.get()[0];
  img_size[1] = orig_img_dim_.get()[1];
  auto img_size_tensor_result = img_size_tensor.value()->reshapeCustom(
  img_size_tensor_shape, gxf::PrimitiveType::kFloat32,
      gxf::PrimitiveTypeSize(gxf::PrimitiveType::kFloat32),
  gxf::Unexpected{GXF_UNINITIALIZED_VALUE}, gxf::MemoryStorageType::kDevice, allocator_.get());  
  if(!img_size_tensor_result){
    GXF_LOG_ERROR("Error allocating tensor for image size tensor.");
    return gxf::ToResultCode(img_size_tensor_result);
  }
  cudaMemcpy(img_size_tensor.value()->pointer(),img_size.data(), 2*sizeof(float),cudaMemcpyHostToDevice);

  out_points_->publish(std::move(out_message.value()));
  return GXF_SUCCESS;
}

void SegmentAnythingPromptProcessor::detectionToSAMPrompt(
    std::vector<nvidia::isaac_ros::Detection2D>& detections,
    std::vector<float>& prompt_vec,
    std::vector<float>& label_vec
    ) {
      uint32_t orig_width = orig_img_dim_.get()[1];
      uint32_t orig_height = orig_img_dim_.get()[0];
      float width_scale = float(resized_width_) / orig_width;
      float height_scale = float(resized_height_) / orig_height;
      for (int idx=0; idx<detections.size(); idx++) {
        nvidia::isaac_ros::Detection2D bbox = detections[idx];
        if (prompt_type_value_ == PromptType::kBbox) {
          float top_left_corner_x = (bbox.center_x - (bbox.size_x / 2.0)) * width_scale;
          float top_left_corner_y = (bbox.center_y - (bbox.size_y / 2.0)) * height_scale;
          float bottom_right_corner_x = (bbox.center_x + (bbox.size_x / 2.0)) * width_scale;
          float bottom_right_corner_y = (bbox.center_y + (bbox.size_y / 2.0)) * height_scale;
          prompt_vec.push_back(top_left_corner_x);
          prompt_vec.push_back(top_left_corner_y);
          prompt_vec.push_back(bottom_right_corner_x);
          prompt_vec.push_back(bottom_right_corner_y);
          label_vec.push_back(2.0);
          label_vec.push_back(3.0);
        } else {
          prompt_vec.push_back(bbox.center_x * width_scale);
          prompt_vec.push_back(bbox.center_y * height_scale);
          label_vec.push_back(1.0);
        }
      }
      // Add the padding point
      if (prompt_type_value_ == PromptType::kPoint) {
        prompt_vec.push_back(0.0);
        prompt_vec.push_back(0.0);
        label_vec.push_back(-1);
      }
  }
gxf_result_t SegmentAnythingPromptProcessor::stop() {
  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
