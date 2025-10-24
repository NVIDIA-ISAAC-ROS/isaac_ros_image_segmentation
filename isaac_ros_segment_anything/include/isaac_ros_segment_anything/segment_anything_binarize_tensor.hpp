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

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

// Binarizes tensor data on GPU by setting all non-zero values to 255
void BinarizeTensorOnGPU(uint8_t * tensor, size_t size, cudaStream_t stream);

// Struct to hold bounding box coordinates
struct BoundingBox
{
  int min_x = 0;
  int min_y = 0;
  int max_x = 0;
  int max_y = 0;
};

// Find the bounding box of non-zero values in a binary mask
void FindBoundingBoxOnGPU(
  const uint8_t * input_data, int width, int height,
  BoundingBox * output_bbox, cudaStream_t stream);

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia
