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
#include "segmentation_postprocessor.cu.hpp"

namespace nvidia {
namespace isaac_ros {

__forceinline__ __device__ uint32_t hwc_to_index(Shape shape, uint32_t y, uint32_t x, uint32_t c) {
  return (y * shape.width + x) * shape.channels + c;
}

__forceinline__ __device__ uint32_t nchw_to_index(Shape shape, uint32_t y, uint32_t x, uint32_t c) {
  return (c * shape.height + y) * shape.width + x;
}

template <DataFormat>
__forceinline__ __device__ uint32_t data_format_to_index(Shape shape, uint32_t y, uint32_t x,
                                                         uint32_t c) {}

template <>
__forceinline__ __device__ uint32_t data_format_to_index<DataFormat::kHWC>(Shape shape, uint32_t y,
                                                                           uint32_t x, uint32_t c) {
  return hwc_to_index(shape, y, x, c);
}

template <>
__forceinline__ __device__ uint32_t data_format_to_index<DataFormat::kNHWC>(Shape shape, uint32_t y,
                                                                            uint32_t x,
                                                                            uint32_t c) {
  return hwc_to_index(shape, y, x, c);
}

template <>
__forceinline__ __device__ uint32_t data_format_to_index<DataFormat::kNCHW>(Shape shape, uint32_t y,
                                                                            uint32_t x,
                                                                            uint32_t c) {
  return nchw_to_index(shape, y, x, c);
}

__forceinline__ __device__ uint32_t hw1_to_index(Shape shape, uint32_t y, uint32_t x) {
  return y * shape.width + x;
}

template <DataFormat data_format>
__global__ void postprocessing_kernel(Shape shape, const float* input, output_type_t* output,
                                      NetworkOutputType network_output_type) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= shape.width) || (y >= shape.height)) { return; }

  float max_value = 0.0f;
  uint8_t max_index = 0;

  switch (network_output_type) {
    case NetworkOutputType::kSigmoid: {
      const float value = input[data_format_to_index<data_format>(shape, y, x, 0)];
      max_index = value >= 0.5f ? 1 : 0;
    } break;
    case NetworkOutputType::kSoftmax: {
      for (uint32_t c = 0; c < shape.channels; c++) {
        const float value = input[data_format_to_index<data_format>(shape, y, x, c)];
        if (value > max_value) {
          max_value = value;
          max_index = c;
        }
      }
    } break;
  }

  output[hw1_to_index(shape, y, x)] = max_index;
}

template <DataFormat data_format>
__global__ void CopyTensorDataCuda(Shape shape, const int32_t* input, output_type_t* output,
                                   NetworkOutputType network_output_type) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= shape.width) || (y >= shape.height)) { return; }

  const int32_t class_id{input[data_format_to_index<data_format>(shape, y, x, 0)]};
  output[hw1_to_index(shape, y, x)] = static_cast<output_type_t>(class_id);
}

uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

void cuda_postprocess(NetworkOutputType network_output_type, DataFormat data_format, Shape shape,
                      const float* input, output_type_t* output, cudaStream_t stream) {
  dim3 block(32, 32, 1);
  dim3 grid(ceil_div(shape.width, block.x), ceil_div(shape.height, block.y), 1);
  switch (data_format) {
    case DataFormat::kNCHW:
      postprocessing_kernel<DataFormat::kNCHW>
          <<<grid, block, 0, stream>>>(shape, input, output, network_output_type);
      break;
    case DataFormat::kHWC:
      postprocessing_kernel<DataFormat::kHWC>
          <<<grid, block, 0, stream>>>(shape, input, output, network_output_type);
      break;
    case DataFormat::kNHWC:
      postprocessing_kernel<DataFormat::kNHWC>
          <<<grid, block, 0, stream>>>(shape, input, output, network_output_type);
      break;
  }
}

void CopyTensorData(NetworkOutputType network_output_type, DataFormat data_format, Shape shape,
                    const int32_t* input, output_type_t* output, cudaStream_t stream) {
  if (network_output_type != NetworkOutputType::kArgmax) { return; }
  dim3 block(32, 32, 1);
  dim3 grid(ceil_div(shape.width, block.x), ceil_div(shape.height, block.y), 1);
  switch (data_format) {
    case DataFormat::kNCHW:
      CopyTensorDataCuda<DataFormat::kNCHW>
          <<<grid, block, 0, stream>>>(shape, input, output, network_output_type);
      break;
    case DataFormat::kHWC:
      CopyTensorDataCuda<DataFormat::kHWC>
          <<<grid, block, 0, stream>>>(shape, input, output, network_output_type);
      break;
    case DataFormat::kNHWC:
      CopyTensorDataCuda<DataFormat::kNHWC>
          <<<grid, block, 0, stream>>>(shape, input, output, network_output_type);
      break;
  }
}

}  // namespace isaac_ros
}  // namespace nvidia
