// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENT_ANYTHING_BBOX_PROCESSOR_HPP_
#define NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENT_ANYTHING_BBOX_PROCESSOR_HPP_

#include <string>

#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"
#include "detection2_d_array_message.hpp"
#include "isaac_ros_nitros_detection2_d_array_type/nitros_detection2_d_array.hpp"

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"

namespace nvidia {
namespace isaac_ros {

enum class PromptType {
  kBbox,
  kPoint
};

class SegmentAnythingPromptProcessor : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  void detectionToSAMPrompt(
    std::vector<nvidia::isaac_ros::Detection2D>& detections,
    std::vector<float>& prompt_vec,
    std::vector<float>& label_vec
    );
 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> in_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> out_points_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<std::vector<int32_t>> orig_img_dim_;
  gxf::Parameter<int32_t> max_batch_size_;
  gxf::Parameter<std::string> prompt_type_name_;
  gxf::Parameter<bool> has_input_mask_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;
  // CUDA stream variables
  gxf::Handle<gxf::CudaStream> cuda_stream_handle_;
  cudaStream_t cuda_stream_ = 0;

  const uint16_t IMAGE_WIDTH_ = 1024;
  const uint16_t IMAGE_HEIGHT_ = 1024;
  uint16_t resized_width_;
  uint16_t resized_height_;
  uint32_t num_points_ = 2;
  PromptType prompt_type_value_;
};
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENT_ANYTHING_BBOX_PROCESSOR_HPP_
