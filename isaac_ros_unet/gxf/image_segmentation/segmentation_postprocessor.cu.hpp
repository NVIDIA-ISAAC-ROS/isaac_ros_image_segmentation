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
#ifndef NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_CU_HPP_
#define NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_CU_HPP_

#include <cstdint>
#include <limits>

#include "cuda.h"
#include "cuda_runtime.h"

namespace nvidia {
namespace isaac_ros {

struct Shape {
  int32_t height;
  int32_t width;
  int32_t channels;
};

enum class NetworkOutputType {
  kArgmax,
  kSigmoid,
  kSoftmax,
};

enum class DataFormat {
  kNCHW,
  kHWC,
  kNHWC,
};

typedef uint8_t output_type_t;

static constexpr int64_t kMaxChannelCount = std::numeric_limits<output_type_t>::max();

void cuda_postprocess(NetworkOutputType network_output_type, DataFormat data_format, Shape shape,
                      const float* input, output_type_t* output, cudaStream_t stream);

void CopyTensorData(NetworkOutputType network_output_type, DataFormat data_format, Shape shape,
                    const int32_t* input, output_type_t* output, cudaStream_t stream);

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_CU_HPP_
