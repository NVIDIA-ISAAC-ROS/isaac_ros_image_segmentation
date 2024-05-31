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
#include "segment_anything_postprocessor.cu.hpp"

namespace nvidia {
namespace isaac_ros {


__forceinline__ __device__ uint32_t nchw_to_index(Shape shape, uint32_t y, uint32_t x, uint32_t c, uint32_t n) {
  return n * shape.channels * shape.width * shape.height + (c * shape.height + y) * shape.width + x;
}

__global__ void postprocessing_kernel(Shape shape, const float* input, output_type_t* output) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= shape.width) || (y >= shape.height)) { return; }
  // For each batch output check if there is a detection.
  for (uint32_t n = 0; n < shape.batch_size ; n++) {
    for (uint32_t c = 0; c < shape.channels; c++) {
      uint32_t idx = nchw_to_index(shape, y, x, c, n);
      const float value = input[idx];
      // If value is greater than 0 then that pixel is of interest
      if (value > 0.0) {
        output[idx] = 1;
      }
      else {
        output[idx] = 0;
      }
    }
  }
}

uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

void cuda_postprocess(Shape shape, const float* input, output_type_t* output, cudaStream_t stream) {
  dim3 block(32, 32, 1);
  dim3 grid(ceil_div(shape.width, block.x), ceil_div(shape.height, block.y), 1);
  
  postprocessing_kernel
      <<<grid, block, 0, stream>>>(shape, input, output);
      
}

}  // namespace isaac_ros
}  // namespace nvidia
