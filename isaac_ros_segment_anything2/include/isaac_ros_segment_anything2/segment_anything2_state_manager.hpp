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

#ifndef ISAAC_ROS_SEGMENT_ANYTHING2__SEGMENT_ANYTHING2_STATE_MANAGER_HPP_
#define ISAAC_ROS_SEGMENT_ANYTHING2__SEGMENT_ANYTHING2_STATE_MANAGER_HPP_

#include <cuda_runtime.h>
#include <unordered_map>
#include <optional>
#include <string>
#include <memory>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <iostream>
#include <utility>
#include <variant>
#include <tuple>
#include <array>
#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything2
{

// Enum to track how an object was initially defined
enum class SAM2PromptType
{
  BBOX,     // Object defined by bounding box
  POINTS,   // Object defined by points
  UNKNOWN   // Not set or unknown
};


enum class TimestampType
{
  MEMORY_UPDATE,
  OBJECT_UPDATE,
  LAST_FRAME_WITH_PROMPT
};

struct BBox
{
  float top_left_x;
  float top_left_y;
  float bottom_right_x;
  float bottom_right_y;
  BBox(float top_left_x, float top_left_y, float bottom_right_x, float bottom_right_y)
  : top_left_x(top_left_x), top_left_y(top_left_y),
    bottom_right_x(bottom_right_x), bottom_right_y(bottom_right_y)
  {}
  std::array<float, 4> toArray() const
  {
    return {top_left_x, top_left_y, bottom_right_x, bottom_right_y};
  }
};

struct Point
{
  float x;
  float y;
  int label;
  Point(float x, float y, int label)
  : x(x), y(y), label(label)
  {}
};

struct SAM2BufferData
{
  float * mask_mem = nullptr;
  float * obj_ptr_mem = nullptr;
  float * bbox_coords = nullptr;
  float * point_coords = nullptr;
  int * point_labels = nullptr;
  int64_t * permutation = nullptr;
  int batch_size = 0;
  int num_points = 0;
  int num_bboxes = 0;
};

// SAM2ObjectMemory encapsulates the tensor memories for a single tracked object
class SAM2Object
{
public:
  SAM2Object(const std::string & object_id, SAM2PromptType prompt_type);
  ~SAM2Object();

  // Initialize memory with zeros
  void allocateMemory(cudaStream_t stream);
  void deallocateMemory();
  // Access memory buffers
  float * getMaskMemory() {return mask_memory_;}
  float * getObjPtrMemory() {return obj_ptr_memory_;}
  int32_t getMaskIdx() {return mask_idx_;}
  int64_t getObjectUpdateTimestamp() {return object_update_timestamp_;}
  int64_t getMemoryUpdateTimestamp() {return memory_update_timestamp_;}
  int64_t getLastFrameWithPromptTimestamp() {return last_frame_with_prompt_timestamp_;}
  bool updateTimestamp(TimestampType type, int64_t timestamp);
  // Update memory for this object
  void updateMemory(
    const float * mask_memory,
    const float * mask_pos_enc,
    const float * obj_ptr,
    cudaStream_t stream,
    bool update_condition_memory = false);

  // Store bounding box used to define this object
  void setInitialBoundingBox(const BBox & bbox);

  // Store points used to define this object
  bool setPoints(
    const std::vector<float> & points,
    const std::vector<int> & labels);

  // Get prompt type
  SAM2PromptType getPromptType() const {return prompt_type_;}
  // Get prompt data (returns either BBox or vector<Point> based on prompt type)
  std::variant<BBox, std::vector<Point>> getPromptData() const;

private:
  float * mask_memory_;  // Shape [4, mem_dim, mem_dim, mem_dim]
  float * obj_ptr_memory_;  // Shape [2, 256]

  std::string object_id_;

  // Prompt type for this object
  SAM2PromptType prompt_type_ = SAM2PromptType::UNKNOWN;

  // Bbox data if defined by bbox
  std::optional<BBox> bbox_;
  // Points data if defined by points
  std::vector<Point> points_;

  // Timestamp of the last update to the object
  // Used to determine when was the last time the object bbox/points/or mask index was updated
  // We only update the memories if the memory timestamp is greater than the object update
  // timestamp. Let's say the object for which mask index was 2 was deleted,
  // i.e.  object whose index was expected at index 3 is now updated to index 2 in state manager
  // but if an old memory msg comes where this information wasn't passed, we can use timestamp to
  // determine if we should update the memory or not.
  int64_t object_update_timestamp_ = 0;
  int64_t memory_update_timestamp_ = 0;

  // Last timestamp when we sent a prompt for this object
  // We continue to send objectprompts with an image unless we have received a memory for
  // that object but for initial frames we get late response from the model. Once we have
  // received a memory for that object we can ignore other frames up until
  // last_frame_with_prompt_timestamp_
  int64_t last_frame_with_prompt_timestamp_ = 0;
  int32_t mask_idx_;
};

class SegmentAnything2DataEncoderNode;
// SAM2StateManager manages the tracking state of all objects
class SAM2StateManager
{
public:
  explicit SAM2StateManager(SegmentAnything2DataEncoderNode * node);

  static constexpr int kMaxPointsPerObject = 5;
  // Add a new object to track with bbox
  // pair of object id and mask index
  std::vector<std::string> addObjects(
    const std::vector<std::string> & object_ids_bbox,
    const std::vector<std::string> & object_ids_points,
    const std::vector<BBox> & bbox_coords,
    const std::vector<std::vector<float>> & points,
    const std::vector<std::vector<int>> & labels,
    int64_t timestamp, cudaStream_t stream);
  // Update memories for all objects after inference
  void updateAllMemories(
    const float * mask_memories,
    const float * mask_pos_enc,
    const float * obj_ptr_memories,
    const float * object_score_logits,
    cudaStream_t stream,
    int64_t batch_size,
    int64_t memory_timestamp);

  int64_t getNumberOfObjects() const {return objects_.size();}

  SAM2BufferData getBuffers(
    cudaStream_t stream,
    const int64_t timestamp);

  bool removeObject(const std::string & obj_id, const int64_t timestamp);

  void getObjectIdsToOutputMaskIdx(
    std::vector<std::string> & object_ids,
    std::vector<int32_t> & output_mask_idx);

  std::vector<std::string> getAllObjectIds();

private:
  SegmentAnything2DataEncoderNode * node_;
  // Map object IDs to objects
  std::unordered_map<std::string, std::shared_ptr<SAM2Object>> objects_;

  // Mapping from object IDs to output mask indices
  std::unordered_map<std::string, int64_t> obj_ids_to_output_mask_idx_;

  // vector of object ids in the order they will be processed by the model
  std::vector<std::string> output_mask_idx_to_obj_id_;
  std::vector<std::string> new_bbox_ids_;
  std::vector<std::string> new_point_ids_;

  int64_t last_image_received_timestamp_ = 0;

  std::tuple<float *, int *> getPointBuffer(
    cudaStream_t stream, int64_t timestamp);
  float * getBboxBuffer(
    cudaStream_t stream, int64_t timestamp);
  int64_t * getPermutationBuffer(cudaStream_t stream);
  // Add a new object to track (generic version)
  bool addObject(
    const std::string & obj_id,
    const SAM2PromptType prompt_type,
    int64_t timestamp,
    cudaStream_t stream,
    const std::optional<BBox> & bbox = std::nullopt,
    const std::optional<std::vector<float>> & points = std::nullopt,
    const std::optional<std::vector<int>> & labels = std::nullopt);

  void removeObjectMaskMapping(const std::string & obj_id, const int64_t timestamp);
};

}  // namespace segment_anything2
}  // namespace isaac_ros
}  // namespace nvidia
#endif  // ISAAC_ROS_SEGMENT_ANYTHING2__SEGMENT_ANYTHING2_STATE_MANAGER_HPP_
