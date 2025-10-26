// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_segment_anything/segment_anything_binarize_tensor.hpp"

#include <algorithm>
#include <limits>

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

namespace
{

constexpr size_t kBlockSize = 256;

}

__global__ void BinarizeTensorKernel(uint8_t * tensor, const size_t size)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    tensor[idx] = (tensor[idx] > 0) ? 255 : 0;
  }
}

void BinarizeTensorOnGPU(uint8_t * tensor, const size_t size, cudaStream_t stream)
{
  const int num_blocks = (size + kBlockSize - 1) / kBlockSize;
  BinarizeTensorKernel <<< num_blocks, kBlockSize, 0, stream >>> (tensor, size);
}

// CUDA kernel to find min/max coordinates of non-zero values
__global__ void FindBoundingBoxKernel(
  const uint8_t * data, int width, int height,
  int * min_x, int * min_y, int * max_x, int * max_y)
{
  extern __shared__ int shared_data[];
  // Set up shared memory for reduction
  int * s_min_x = &shared_data[0];
  int * s_min_y = &shared_data[blockDim.x];
  int * s_max_x = &shared_data[2 * blockDim.x];
  int * s_max_y = &shared_data[3 * blockDim.x];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize shared memory with extreme values
  s_min_x[tid] = width;
  s_min_y[tid] = height;
  s_max_x[tid] = -1;
  s_max_y[tid] = -1;

  // Process multiple pixels per thread if needed
  for (int i = gid; i < width * height; i += blockDim.x * gridDim.x) {
    if (data[i] > 0) {  // Non-zero pixel found
      const int x = i % width;
      const int y = i / width;

      s_min_x[tid] = min(s_min_x[tid], x);
      s_min_y[tid] = min(s_min_y[tid], y);
      s_max_x[tid] = max(s_max_x[tid], x);
      s_max_y[tid] = max(s_max_y[tid], y);
    }
  }

  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_min_x[tid] = min(s_min_x[tid], s_min_x[tid + s]);
      s_min_y[tid] = min(s_min_y[tid], s_min_y[tid + s]);
      s_max_x[tid] = max(s_max_x[tid], s_max_x[tid + s]);
      s_max_y[tid] = max(s_max_y[tid], s_max_y[tid + s]);
    }
    __syncthreads();
  }

  // Write results to global memory
  if (tid == 0) {
    atomicMin(min_x, s_min_x[0]);
    atomicMin(min_y, s_min_y[0]);
    atomicMax(max_x, s_max_x[0]);
    atomicMax(max_y, s_max_y[0]);
  }
}

void FindBoundingBoxOnGPU(
  const uint8_t * input_data, int width, int height, BoundingBox * output_bbox, cudaStream_t stream)
{
  // Initialize device memory for results
  int * d_min_x;
  int * d_min_y;
  int * d_max_x;
  int * d_max_y;

  // Allocate device memory
  cudaMallocAsync(&d_min_x, sizeof(int), stream);
  cudaMallocAsync(&d_min_y, sizeof(int), stream);
  cudaMallocAsync(&d_max_x, sizeof(int), stream);
  cudaMallocAsync(&d_max_y, sizeof(int), stream);

  // Initialize with extreme values
  int h_min_x = width;
  int h_min_y = height;
  int h_max_x = -1;
  int h_max_y = -1;

  cudaMemcpyAsync(d_min_x, &h_min_x, sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_min_y, &h_min_y, sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_max_x, &h_max_x, sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_max_y, &h_max_y, sizeof(int), cudaMemcpyHostToDevice, stream);

  // Launch kernel
  const int num_blocks = (width * height + kBlockSize - 1) / kBlockSize;
  const int shared_mem_size = 4 * kBlockSize * sizeof(int);

  FindBoundingBoxKernel <<< num_blocks, kBlockSize, shared_mem_size, stream >>> (
    input_data, width, height, d_min_x, d_min_y, d_max_x, d_max_y);

  // Copy results back to host
  cudaMemcpyAsync(&output_bbox->min_x, d_min_x, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&output_bbox->min_y, d_min_y, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&output_bbox->max_x, d_max_x, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&output_bbox->max_y, d_max_y, sizeof(int), cudaMemcpyDeviceToHost, stream);

  // Free device memory
  cudaFreeAsync(d_min_x, stream);
  cudaFreeAsync(d_min_y, stream);
  cudaFreeAsync(d_max_x, stream);
  cudaFreeAsync(d_max_y, stream);
}

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia
