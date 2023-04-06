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
#ifndef NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_HPP_
#define NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_HPP_

#include <string>

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "segmentation_postprocessor.cu.hpp"

namespace nvidia {
namespace isaac_ros {

class SegmentationPostprocessor : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  NetworkOutputType network_output_type_value_;
  DataFormat data_format_value_;

  gxf::Parameter<gxf::Handle<gxf::Receiver>> in_;
  gxf::Parameter<std::string> in_tensor_name_;
  gxf::Parameter<std::string> network_output_type_;
  gxf::Parameter<std::string> data_format_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> out_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;

  gxf::Handle<gxf::CudaStream> stream_;
};
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_HPP_
