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
#include <string>
#include <utility>

#include "cuda.h"
#include "cuda_runtime.h"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"
#include "segment_anything_msg_compositor.hpp"

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

gxf_result_t SegmentAnythingMsgCompositor::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(inputs_, "inputs", "Input", "List of input receivers");
  result &= registrar->parameter(output_, "output", "Output", "Output Transmitter");
  return gxf::ToResultCode(result);
}

gxf_result_t SegmentAnythingMsgCompositor::start() {
  return GXF_SUCCESS;
}

gxf_result_t SegmentAnythingMsgCompositor::tick() {
  // Process input message
  auto out_message = gxf::Entity::New(context());
  bool assign_ts = false;
  bool is_data_valid = true;
  for (const auto & rx : inputs_.get()) {
    const auto in_message = rx->receive();

    if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }
    // Look for invalid_frame tensor, it means the current input has no corresponding prompt.
    auto maybe_invalid_data = in_message.value().get<gxf::Tensor>("invalid_frame");
    if (maybe_invalid_data) {
      is_data_valid = false;
    }
    auto maybe_tensor_list = in_message.value().findAll<gxf::Tensor>();
    if (!maybe_tensor_list) { return gxf::ToResultCode(maybe_tensor_list); }
    auto tensor_list = maybe_tensor_list.value();

    for (int idx = 0; idx<tensor_list.size();idx++) {
        auto maybe_in_tensor = tensor_list[idx];
        auto in_tensor = maybe_in_tensor.value();

        auto out_tensor = out_message.value().add<gxf::Tensor>(in_tensor.name());
        if (!out_tensor) { return gxf::ToResultCode(out_tensor); }
        
        out_tensor.value()->wrapMemoryBuffer(in_tensor->shape(),
        in_tensor->element_type(),gxf::PrimitiveTypeSize(in_tensor->element_type()),
        gxf::Unexpected{GXF_UNINITIALIZED_VALUE},in_tensor->move_buffer());
    }
    if(!assign_ts){
      auto maybe_added_timestamp = AddInputTimestampToOutput(out_message.value(), in_message.value());
      if(maybe_added_timestamp){
        assign_ts = true;
      }
    }
  }
  // Only publish if the data is valid.
  if (is_data_valid) {
    output_->publish(std::move(out_message.value()));
  }
  return GXF_SUCCESS;
}

gxf_result_t SegmentAnythingMsgCompositor::stop() {
  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
