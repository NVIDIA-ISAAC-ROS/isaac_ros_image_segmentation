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
#ifndef NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_MASK_COLORIZER_HPP_
#define NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_MASK_COLORIZER_HPP_

#include <string>
#include <vector>

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"
#include "segmentation_mask_colorizer.cu.hpp"

namespace nvidia {
namespace isaac_ros {

class SegmentationMaskColorizer : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> input_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> colored_segmentation_output_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<std::vector<int64_t>> color_palette_vec_;
  gxf::Parameter<std::string> color_segmentation_mask_encoding_str_;

  ArrayView<int64_t> color_palette_;
  ColorImageEncodings color_segmentation_mask_encoding_;

  gxf::Expected<void> generateColorizedSegmentationMask(gxf::Entity& output, gxf::Entity input,
                                                        gxf::Handle<gxf::CudaStream> stream);

  gxf::Expected<void> createColorSegmentationMask(
      gxf::Handle<gxf::VideoBuffer> colored_segmentation_mask,
      gxf::Handle<gxf::VideoBuffer> raw_segmentation_mask);
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_MASK_COLORIZER_HPP_
