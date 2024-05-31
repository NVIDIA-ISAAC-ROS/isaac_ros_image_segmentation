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
#include "segmentation_mask_colorizer.cu.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace nvidia {
namespace isaac_ros {

namespace {

__device__ inline uint32_t GetFlattenedIndex(const uint32_t v, const uint32_t u, const uint32_t c,
                                             const uint32_t width, const uint32_t channels) {
  return v * width * channels + u * channels + c;
}

__device__ inline uint32_t GetMonoIndex(const uint32_t v, const uint32_t u, const uint32_t width) {
  return GetFlattenedIndex(v, u, 0, width, 1);
}

template <ColorImageEncodings e>
__device__ inline uint32_t GetRedIndex(const uint32_t v, const uint32_t u, const uint32_t width);

template <>
__device__ inline uint32_t GetRedIndex<ColorImageEncodings::kRGB8>(const uint32_t v,
                                                                   const uint32_t u,
                                                                   const uint32_t width) {
  return GetFlattenedIndex(v, u, 0, width, 3);
}

template <>
__device__ inline uint32_t GetRedIndex<ColorImageEncodings::kBGR8>(const uint32_t v,
                                                                   const uint32_t u,
                                                                   const uint32_t width) {
  return GetFlattenedIndex(v, u, 2, width, 3);
}

template <ColorImageEncodings e>
__device__ inline uint32_t GetGreenIndex(const uint32_t v, const uint32_t u, const uint32_t width) {
  static_assert(e == ColorImageEncodings::kBGR8 || e == ColorImageEncodings::kRGB8,
                "Error: received invalid encoding!");
  return GetFlattenedIndex(v, u, 1, width, 3);
}

template <ColorImageEncodings e>
__device__ inline uint32_t GetBlueIndex(const uint32_t v, const uint32_t u, const uint32_t width);

template <>
__device__ inline uint32_t GetBlueIndex<ColorImageEncodings::kRGB8>(const uint32_t v,
                                                                    const uint32_t u,
                                                                    const uint32_t width) {
  return GetFlattenedIndex(v, u, 2, width, 3);
}

template <>
__device__ inline uint32_t GetBlueIndex<ColorImageEncodings::kBGR8>(const uint32_t v,
                                                                    const uint32_t u,
                                                                    const uint32_t width) {
  return GetFlattenedIndex(v, u, 0, width, 3);
}

template <ColorImageEncodings e>
__global__ void ColorizeSegmentationMaskImpl(uint8_t* colored_segmentation_mask,
                                             const uint32_t width, const uint32_t height,
                                             const uint8_t* raw_segmentation_mask,
                                             const int64_t* class_to_color,
                                             const size_t class_to_color_size) {
  uint32_t u_idx{blockIdx.x * blockDim.x + threadIdx.x};
  uint32_t u_stride{gridDim.x * blockDim.x};

  uint32_t v_idx{blockIdx.y * blockDim.y + threadIdx.y};
  uint32_t v_stride{gridDim.y * blockDim.y};

  for (uint32_t v = v_idx; v < height; v += v_stride) {
    for (uint32_t u = u_idx; u < width; u += u_stride) {
      const uint8_t class_id{raw_segmentation_mask[GetMonoIndex(v, u, width)]};
      const int64_t selected_color{class_to_color[class_id % class_to_color_size]};
      colored_segmentation_mask[GetRedIndex<e>(v, u, width)] = (selected_color >> 16) & 0xFF;
      colored_segmentation_mask[GetGreenIndex<e>(v, u, width)] = (selected_color >> 8) & 0xFF;
      colored_segmentation_mask[GetBlueIndex<e>(v, u, width)] = selected_color & 0xFF;
    }
  }
}

}  // namespace

void ColorizeSegmentationMask(uint8_t* colored_segmentation_mask, const uint32_t width,
                              const uint32_t height, const ColorImageEncodings image_encoding,
                              const uint8_t* raw_segmentation_mask,
                              const ArrayView<int64_t>& color_palette, cudaStream_t stream) {
  dim3 threads_per_block{32, 32, 1};
  dim3 blocks{(width + threads_per_block.x - 1) / threads_per_block.x,
              (height + threads_per_block.y - 1) / threads_per_block.y, 1};
  const int64_t* class_to_color = color_palette.data.get();
  const size_t class_to_color_size = color_palette.size;

  switch (image_encoding) {
    case ColorImageEncodings::kRGB8:
      ColorizeSegmentationMaskImpl<ColorImageEncodings::kRGB8>
          <<<blocks, threads_per_block, 0, stream>>>(colored_segmentation_mask, width, height,
                                                     raw_segmentation_mask, class_to_color,
                                                     class_to_color_size);
      break;
    case ColorImageEncodings::kBGR8:
      ColorizeSegmentationMaskImpl<ColorImageEncodings::kBGR8>
          <<<blocks, threads_per_block, 0, stream>>>(colored_segmentation_mask, width, height,
                                                     raw_segmentation_mask, class_to_color,
                                                     class_to_color_size);
      break;
  }
}

}  // namespace isaac_ros
}  // namespace nvidia
