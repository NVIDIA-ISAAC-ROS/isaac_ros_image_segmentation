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
#ifndef NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENT_ANYTHING_POSTPROCESSOR_CU_HPP_
#define NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENT_ANYTHING_POSTPROCESSOR_CU_HPP_

#include <cstdint>
#include <limits>

#include "cuda.h"
#include "cuda_runtime.h"

namespace nvidia {
namespace isaac_ros {

struct Shape {
  int32_t batch_size;
  int32_t height;
  int32_t width;
  int32_t channels;
};

typedef uint8_t output_type_t;

static constexpr int32_t kExpectedChannelCount = 1;

void cuda_postprocess(Shape shape, const float* input, output_type_t* output, cudaStream_t stream);
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_CU_HPP_
