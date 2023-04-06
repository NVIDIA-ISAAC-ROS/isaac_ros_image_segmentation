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
#ifndef NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_MASK_COLORIZER_CU_HPP_
#define NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_MASK_COLORIZER_CU_HPP_

#include <cstdint>
#include <memory>

#include "cuda.h"
#include "cuda_runtime.h"

namespace nvidia {
namespace isaac_ros {

template <typename T>
struct ArrayView {
  std::unique_ptr<T[], decltype(&cudaFree)> data{nullptr, cudaFree};
  std::size_t size;
};

enum class ColorImageEncodings { kRGB8, kBGR8 };

void ColorizeSegmentationMask(uint8_t* colored_segmentation_mask, uint32_t width, uint32_t height,
                              ColorImageEncodings image_encoding,
                              const uint8_t* raw_segmentation_mask,
                              const ArrayView<int64_t>& color_palette, cudaStream_t stream);

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_MASK_COLORIZER_CU_HPP_
