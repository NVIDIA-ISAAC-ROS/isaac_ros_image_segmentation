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

#include "isaac_ros_segment_anything2/segment_anything2_state_manager.hpp"
#include "isaac_ros_segment_anything2/segment_anything2_data_encoder_node.hpp"
#include <vector>
#include <iostream>
#include <variant>
#include <optional>

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything2
{

namespace
{
// Used to represent the shape and size of memory tensors
// mask_mem tensor shape: [4, 64, 64, 64] with type float
// obj_ptr tensor shape: [2, 256] with type float
constexpr int64_t kIdxCountOfMaskMem = 4;
constexpr int64_t kIdxCountOfObjPtrMem = 2;

constexpr int64_t kMaskMemPerIdxElements = 64 * 64 * 64;
constexpr int64_t kObjPtrMemPerIdxElements = 256;

constexpr int64_t kMaskMemTotalElements = kIdxCountOfMaskMem * kMaskMemPerIdxElements;
constexpr int64_t kObjPtrMemTotalElements =
  kIdxCountOfObjPtrMem * kObjPtrMemPerIdxElements;

constexpr int64_t kMaskMemSize = kMaskMemTotalElements * sizeof(float);
constexpr int64_t kObjPtrMemSize = kObjPtrMemTotalElements * sizeof(float);

// Utility functions for index calculations
inline int getPointCoordXIndex(int obj_idx, int point_idx)
{
  return obj_idx * SAM2StateManager::kMaxPointsPerObject * 2 + point_idx * 2;
}

inline int getPointCoordYIndex(int obj_idx, int point_idx)
{
  return obj_idx * SAM2StateManager::kMaxPointsPerObject * 2 + point_idx * 2 + 1;
}

inline int getPointLabelIndex(int obj_idx, int point_idx)
{
  return obj_idx * SAM2StateManager::kMaxPointsPerObject + point_idx;
}

// Utility function for copying memory chunks
void copyMemoryChunk(
  float * dst_mask_mem,
  float * dst_obj_ptr_mem,
  const float * src_mask_mem,
  const float * src_mask_pos_enc,
  const float * src_obj_ptr_mem,
  const int mask_position_offset,
  const int obj_ptr_position_offset,
  cudaStream_t stream,
  const std::string & operation_name)
{
  // Copy mask_memory
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      dst_mask_mem + (mask_position_offset * kMaskMemPerIdxElements),
      src_mask_mem, kMaskMemPerIdxElements * sizeof(float),
      cudaMemcpyDeviceToDevice, stream),
    (std::string("Failed to copy mask memory for ") + operation_name).c_str());

  // Copy mask_pos_enc
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      dst_mask_mem + ((mask_position_offset + 1) * kMaskMemPerIdxElements),
      src_mask_pos_enc, kMaskMemPerIdxElements * sizeof(float),
      cudaMemcpyDeviceToDevice, stream),
    (std::string("Failed to copy mask_pos_enc for ") + operation_name).c_str());

  // Copy obj_ptr
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      dst_obj_ptr_mem + (obj_ptr_position_offset * kObjPtrMemPerIdxElements),
      src_obj_ptr_mem, kObjPtrMemPerIdxElements * sizeof(float),
      cudaMemcpyDeviceToDevice, stream),
    (std::string("Failed to copy obj_ptr for ") + operation_name).c_str());
}

// Utility functions for CUDA memory management
template<typename T>
T * allocateCudaMemory(
  size_t count, cudaStream_t stream,
  const char * error_msg = "Failed to allocate CUDA memory")
{
  T * ptr;
  CHECK_CUDA_ERROR(
    cudaMallocAsync(&ptr, count * sizeof(T), stream),
    error_msg);
  return ptr;
}

template<typename T>
T * allocateAndInitializeCudaMemory(
  size_t count, T init_value, cudaStream_t stream,
  const char * error_msg = "Failed to allocate CUDA memory")
{
  T * ptr = allocateCudaMemory<T>(count, stream, error_msg);
  CHECK_CUDA_ERROR(cudaMemsetAsync(ptr, init_value, count * sizeof(T), stream), error_msg);
  return ptr;
}

void freeCudaMemory(void * ptr, const char * error_msg = "Failed to free CUDA memory")
{
  if (ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr), error_msg);
  }
}
}  // namespace

// Object
SAM2Object::SAM2Object(const std::string & object_id, SAM2PromptType prompt_type)
: object_id_(object_id), prompt_type_(prompt_type) {}

void SAM2Object::allocateMemory(cudaStream_t stream)
{
  // Create mask memory tensor [4, mem_dim, mem_dim, mem_dim]
  mask_memory_ = allocateAndInitializeCudaMemory<float>(
    kMaskMemTotalElements, 0.0f, stream, "Failed to allocate mask memory");

  obj_ptr_memory_ = allocateAndInitializeCudaMemory<float>(
    kObjPtrMemTotalElements, 0.0f, stream, "Failed to allocate obj_ptr memory");
}

bool SAM2Object::updateTimestamp(TimestampType type, int64_t timestamp)
{
  int64_t * target_timestamp = nullptr;

  switch (type) {
    case TimestampType::MEMORY_UPDATE:
      target_timestamp = &memory_update_timestamp_;
      break;
    case TimestampType::OBJECT_UPDATE:
      target_timestamp = &object_update_timestamp_;
      break;
    case TimestampType::LAST_FRAME_WITH_PROMPT:
      target_timestamp = &last_frame_with_prompt_timestamp_;
      break;
  }

  // Update timestamp if it is greater than the current timestamp
  if (timestamp > *target_timestamp) {
    *target_timestamp = timestamp;
    return true;
  }
  return false;
}

void SAM2Object::updateMemory(
  const float * mask_memory,
  const float * mask_pos_enc,
  const float * obj_ptr,
  cudaStream_t stream,
  bool update_condition_memory)
{
  if (!mask_memory_ || !obj_ptr_memory_) {
    throw std::runtime_error("Mask or obj_ptr_memory is not allocated");
  }

  // Update condition frame memory if requested
  if (update_condition_memory) {
    copyMemoryChunk(
      mask_memory_, obj_ptr_memory_,
      mask_memory, mask_pos_enc, obj_ptr,
      0, 0, stream, "condition frame");
  }

  // Always update last frame memory
  copyMemoryChunk(
    mask_memory_, obj_ptr_memory_,
    mask_memory, mask_pos_enc, obj_ptr,
    2, 1, stream, "last frame");
  return;
}

void SAM2Object::setInitialBoundingBox(const BBox & bbox)
{
  bbox_ = bbox;
}

bool SAM2Object::setPoints(
  const std::vector<float> & points,
  const std::vector<int> & labels)
{
  int32_t num_points = labels.size();
  if (num_points > SAM2StateManager::kMaxPointsPerObject) {
    return false;
  }
  for (int32_t i = 0; i < num_points; i++) {
    Point point = Point(points[i * 2], points[i * 2 + 1], labels[i]);
    points_.push_back(point);
  }
  return true;
}

std::variant<BBox, std::vector<Point>> SAM2Object::getPromptData() const
{
  if (prompt_type_ == SAM2PromptType::BBOX) {
    if (!bbox_.has_value()) {
      throw std::runtime_error("BBox data not available");
    }
    return bbox_.value();
  } else if (prompt_type_ == SAM2PromptType::POINTS) {
    return points_;
  } else {
    throw std::runtime_error("Unknown prompt type");
  }
}

void SAM2Object::deallocateMemory()
{
  freeCudaMemory(mask_memory_, "Failed to free mask memory");
  freeCudaMemory(obj_ptr_memory_, "Failed to free obj_ptr memory");
}

SAM2Object::~SAM2Object()
{
  deallocateMemory();
}
// State Manager
SAM2StateManager::SAM2StateManager(SegmentAnything2DataEncoderNode * node)
: node_(node) {}
// Whenever a object is changed, it should be assigned a new index at the end
void SAM2StateManager::removeObjectMaskMapping(const std::string & obj_id, const int64_t timestamp)
{
  auto output_mask_idx_to_obj_id_it = std::find(
    output_mask_idx_to_obj_id_.begin(),
    output_mask_idx_to_obj_id_.end(),
    obj_id);

  if (output_mask_idx_to_obj_id_it == output_mask_idx_to_obj_id_.end()) {
    RCLCPP_ERROR(node_->get_logger(), "Object not found in output_mask_idx_to_obj_id_");
    return;
  }
  int idx = std::distance(output_mask_idx_to_obj_id_.begin(), output_mask_idx_to_obj_id_it);
  for (int i = idx + 1; i < output_mask_idx_to_obj_id_.size(); i++) {
    std::string obj_id = output_mask_idx_to_obj_id_[i];
    obj_ids_to_output_mask_idx_[obj_id] = i - 1;
    // update the timestamp so old memories are not used since index has changed
    RCLCPP_DEBUG(
      node_->get_logger(),
      "Updating object update timestamp for object with id: %s to timestamp: %ld",
      obj_id.c_str(), timestamp);
    objects_[obj_id]->updateTimestamp(TimestampType::OBJECT_UPDATE, timestamp);
  }
  output_mask_idx_to_obj_id_.erase(output_mask_idx_to_obj_id_it);
  obj_ids_to_output_mask_idx_.erase(obj_id);
  return;
}

bool SAM2StateManager::removeObject(const std::string & obj_id, const int64_t timestamp)
{
  auto object_it = objects_.find(obj_id);
  if (object_it == objects_.end()) {
    RCLCPP_WARN(node_->get_logger(), "Object with id: %s not found", obj_id.c_str());
    return false;
  }
  RCLCPP_INFO(node_->get_logger(), "Removing object with id: %s", obj_id.c_str());
  int64_t new_timestamp = std::max(timestamp, last_image_received_timestamp_ + 1);
  removeObjectMaskMapping(obj_id, new_timestamp);
  auto prompt_type = object_it->second->getPromptType();
  objects_.erase(object_it);
  // Also remove from new_bbox_ids_ or new_point_ids_ if it is present
  if (prompt_type == SAM2PromptType::BBOX) {
    auto new_bbox_id_it = std::find(new_bbox_ids_.begin(), new_bbox_ids_.end(), obj_id);
    if (new_bbox_id_it != new_bbox_ids_.end()) {
      new_bbox_ids_.erase(new_bbox_id_it);
    }
  } else if (prompt_type == SAM2PromptType::POINTS) {
    auto new_point_id_it = std::find(new_point_ids_.begin(), new_point_ids_.end(), obj_id);
    if (new_point_id_it != new_point_ids_.end()) {
      new_point_ids_.erase(new_point_id_it);
    }
  }
  return true;
}

bool SAM2StateManager::addObject(
  const std::string & obj_id,
  const SAM2PromptType prompt_type,
  int64_t timestamp,
  cudaStream_t stream,
  const std::optional<BBox> & bbox,
  const std::optional<std::vector<float>> & points,
  const std::optional<std::vector<int>> & labels)
{
  // Use timestamp whichever is greater.
  // Idea is to make sure that the object is not updated with an older timestamp memory
  timestamp = std::max(timestamp, last_image_received_timestamp_ + 1);
  auto object_it = objects_.find(obj_id);
  // If object already exists, remove it
  if (object_it != objects_.end()) {
    removeObject(obj_id, timestamp);
  }
  objects_[obj_id] = std::make_unique<SAM2Object>(obj_id, prompt_type);
  objects_[obj_id]->allocateMemory(stream);

  int64_t new_index = objects_.size() - 1;
  obj_ids_to_output_mask_idx_[obj_id] = new_index;

  output_mask_idx_to_obj_id_.push_back(obj_id);
  objects_[obj_id]->updateTimestamp(TimestampType::OBJECT_UPDATE, timestamp);

  // Set prompt-specific data based on type
  if (prompt_type == SAM2PromptType::BBOX && bbox.has_value()) {
    objects_[obj_id]->setInitialBoundingBox(bbox.value());
    new_bbox_ids_.push_back(obj_id);
    RCLCPP_INFO(
      node_->get_logger(),
      "Added object with id: %s with bbox top_left: %f %f and bottom_right: %f %f and idx:%d",
      obj_id.c_str(), bbox.value().top_left_x, bbox.value().top_left_y,
      bbox.value().bottom_right_x, bbox.value().bottom_right_y,
      output_mask_idx_to_obj_id_.size() - 1);
  } else if (prompt_type == SAM2PromptType::POINTS && points.has_value() && labels.has_value()) {
    bool result = objects_[obj_id]->setPoints(points.value(), labels.value());
    if (!result) {
      return false;
    }
    new_point_ids_.push_back(obj_id);
    RCLCPP_INFO(
      node_->get_logger(),
      "Added object with id: %s with idx:%d", obj_id.c_str(),
      output_mask_idx_to_obj_id_.size() - 1);
    for (int i = 0; i < points.value().size(); i += 2) {
      RCLCPP_INFO(node_->get_logger(), "Points: %f %f", points.value()[i], points.value()[i + 1]);
    }
  } else {
    throw std::runtime_error("Unknown prompt type, or missing data");
  }
  return true;
}

float * SAM2StateManager::getBboxBuffer(
  cudaStream_t stream, int64_t timestamp)
{
  int32_t new_bboxes = new_bbox_ids_.size();
  // Add a dummy bbox at the end hence +1, it gets filtered out by the model
  // needed for Triton support
  // 4 = number of coordinates per bbox
  float * bbox_coords_buffer = allocateCudaMemory<float>(
    (new_bboxes + 1) * 4, stream, "Failed to allocate bbox coords tensor");
  // If new bboxes are present, fill the buffer with the new bboxes
  // Otherwise, initialize the buffer with zeros
  if (new_bboxes > 0) {
    std::vector<std::array<float, 4>> bbox_coords;
    for (auto obj_id : new_bbox_ids_) {
      auto object = objects_[obj_id];
      auto bbox = object->getPromptData();
      bbox_coords.push_back(std::get<BBox>(bbox).toArray());
      object->updateTimestamp(TimestampType::LAST_FRAME_WITH_PROMPT, timestamp);
      RCLCPP_DEBUG(
        node_->get_logger(),
        "Updating object update timestamp for object with id: %s to timestamp: %ld",
        obj_id.c_str(), timestamp);
    }
    CHECK_CUDA_ERROR(
      cudaMemcpyAsync(
        bbox_coords_buffer, bbox_coords.data(), new_bboxes * 4 * sizeof(float),
        cudaMemcpyHostToDevice, stream), "Failed to copy bbox coords to tensor");
  } else {
    CHECK_CUDA_ERROR(
      cudaMemsetAsync(bbox_coords_buffer, 0, 4 * sizeof(float), stream),
      "Failed to initialize empty bbox coords tensor");
  }
  return bbox_coords_buffer;
}

std::tuple<float *, int *> SAM2StateManager::getPointBuffer(
  cudaStream_t stream,
  int64_t timestamp)
{
  int32_t new_points = new_point_ids_.size();
  // Add a dummy point at the end, it gets filtered out by the model, needed for Triton support
  // 2 = number of coordinates per point
  // we need to allocate memory for the maximum number of points per object
  // which is kMaxPointsPerObject
  float * point_coords_buffer = allocateCudaMemory<float>(
    (new_points + 1) * kMaxPointsPerObject * 2, stream,
    "Failed to allocate point coords tensor");
  int * point_labels_buffer = allocateCudaMemory<int>(
    (new_points + 1) * kMaxPointsPerObject, stream, "Failed to allocate point labels tensor");
  // If new points are present, fill the buffer with the new points
  // Otherwise, initialize the buffer with zeros
  if (new_points > 0) {
    // Creating vectors to copy the data to the CUDA buffers
    // point coords for a object are stored x1,y1,x2,y2,x3,y3,...
    // point labels for a object are stored l1,l2,l3,...
    std::vector<float> point_coords(new_points * kMaxPointsPerObject * 2, 0);
    std::vector<int> point_labels(new_points * kMaxPointsPerObject, -1);
    int obj_idx = 0;
    for (auto obj_id : new_point_ids_) {
      auto object = objects_[obj_id];
      auto points = std::get<std::vector<Point>>(object->getPromptData());
      object->updateTimestamp(TimestampType::LAST_FRAME_WITH_PROMPT, timestamp);
      RCLCPP_DEBUG(
        node_->get_logger(),
        "Updating object update timestamp for object with id: %s to timestamp: %ld",
        obj_id.c_str(), timestamp);
      // Only process the actual points this object has
      for (int32_t i = 0; i < points.size(); i++) {
        point_coords[getPointCoordXIndex(obj_idx, i)] = points[i].x;
        point_coords[getPointCoordYIndex(obj_idx, i)] = points[i].y;
        point_labels[getPointLabelIndex(obj_idx, i)] = points[i].label;
      }
      obj_idx++;
    }
    CHECK_CUDA_ERROR(
      cudaMemcpyAsync(
        point_coords_buffer,
        point_coords.data(), new_points * kMaxPointsPerObject * 2 * sizeof(float),
        cudaMemcpyHostToDevice, stream), "Failed to copy point coords to tensor");
    CHECK_CUDA_ERROR(
      cudaMemcpyAsync(
        point_labels_buffer,
        point_labels.data(), new_points * kMaxPointsPerObject * sizeof(int),
        cudaMemcpyHostToDevice, stream), "Failed to copy point labels to tensor");
  } else {
    CHECK_CUDA_ERROR(
      cudaMemsetAsync(point_coords_buffer, 0, kMaxPointsPerObject * 2 * sizeof(float), stream),
      "Failed to initialize empty point coords tensor");
    CHECK_CUDA_ERROR(
      cudaMemsetAsync(point_labels_buffer, 0, kMaxPointsPerObject * sizeof(int), stream),
      "Failed to initialize empty point labels tensor");
  }
  return std::make_tuple(point_coords_buffer, point_labels_buffer);
}

int64_t * SAM2StateManager::getPermutationBuffer(cudaStream_t stream)
{
  int total_objects = objects_.size();
  std::vector<int64_t> permutation(total_objects);
  int new_bbox_count = new_bbox_ids_.size();
  int new_point_count = new_point_ids_.size();
  int padding_count = total_objects - new_bbox_count - new_point_count;

  // Helper lambda to find object in new IDs list and calculate permutation
  auto calculatePermutation = [](const std::vector<std::string> & new_ids,
      const std::string & object_id,
      int base_offset) -> std::optional<int> {
      auto new_id_it = std::find(new_ids.begin(), new_ids.end(), object_id);
      if (new_id_it != new_ids.end()) {
        int distance = std::distance(new_ids.begin(), new_id_it);
        return base_offset + distance;
      }
      return std::nullopt;
    };

  for (auto & object_map : objects_) {
    auto object_id = object_map.first;
    auto object = object_map.second.get();
    int idx = obj_ids_to_output_mask_idx_[object_id];

    std::optional<int> new_permutation_value;

    if (object->getPromptType() == SAM2PromptType::BBOX) {
      new_permutation_value = calculatePermutation(
        new_bbox_ids_, object_id, padding_count + new_point_count);
    } else {
      new_permutation_value = calculatePermutation(
        new_point_ids_, object_id, padding_count);
    }
    permutation[idx] = new_permutation_value.value_or(idx);
  }

  int64_t * permutation_buffer = allocateCudaMemory<int64_t>(
    total_objects, stream, "Failed to allocate permutation tensor");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      permutation_buffer, permutation.data(),
      total_objects * sizeof(int64_t), cudaMemcpyHostToDevice, stream),
    "Failed to copy permutation data to tensor");
  return permutation_buffer;
}

// Update the function signature and implementation
SAM2BufferData SAM2StateManager::getBuffers(
  cudaStream_t stream,
  const int64_t timestamp)
{
  SAM2BufferData buffer_data;

  buffer_data.batch_size = objects_.size();
  if (buffer_data.batch_size == 0) {
    return buffer_data;
  }

  if (timestamp > last_image_received_timestamp_) {
    last_image_received_timestamp_ = timestamp;
  }

  // Create output buffers
  buffer_data.mask_mem = allocateCudaMemory<float>(
    buffer_data.batch_size * kMaskMemTotalElements, stream,
    "Failed to allocate mask memory buffer");
  buffer_data.obj_ptr_mem = allocateCudaMemory<float>(
    buffer_data.batch_size * kObjPtrMemTotalElements, stream,
    "Failed to allocate obj_ptr memory buffer");

  for (const auto & [obj_id, object] : objects_) {
    int idx = obj_ids_to_output_mask_idx_[obj_id];
    // Get memory handles
    float * source_mask = object->getMaskMemory();
    float * source_obj_ptr = object->getObjPtrMemory();
    // Copy mask memory
    float * dst_mask = buffer_data.mask_mem + (idx * kMaskMemTotalElements);
    CHECK_CUDA_ERROR(
      cudaMemcpyAsync(dst_mask, source_mask, kMaskMemSize, cudaMemcpyDeviceToDevice, stream),
      "Failed to copy mask memory to output buffer");

    float * dst_obj_ptr = buffer_data.obj_ptr_mem + (idx * kObjPtrMemTotalElements);
    CHECK_CUDA_ERROR(
      cudaMemcpyAsync(
        dst_obj_ptr, source_obj_ptr,
        kObjPtrMemSize, cudaMemcpyDeviceToDevice, stream),
      "Failed to copy obj_ptr memory to output buffer");
  }
  std::tie(
    buffer_data.point_coords,
    buffer_data.point_labels
  ) = getPointBuffer(stream, timestamp);
  buffer_data.bbox_coords = getBboxBuffer(stream, timestamp);
  buffer_data.permutation = getPermutationBuffer(stream);
  // +1 for the padded point/bbox
  buffer_data.num_points = new_point_ids_.size() + 1;
  buffer_data.num_bboxes = new_bbox_ids_.size() + 1;
  return buffer_data;
}

void SAM2StateManager::updateAllMemories(
  const float * mask_memories,
  const float * mask_pos_enc,
  const float * obj_ptr_memories,
  const float * object_score_logits,
  cudaStream_t stream,
  int64_t batch_size,
  int64_t memory_timestamp)
{
  std::vector<float> object_scores(batch_size);
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      object_scores.data(), object_score_logits,
      batch_size * sizeof(float),
      cudaMemcpyDeviceToHost, stream),
    "Failed to copy object scores from device");

  int64_t num_objects_in_manager = objects_.size();
  int64_t num_objects = std::min(batch_size, num_objects_in_manager);

  // Synchronize before using the object scores on CPU.
  CHECK_CUDA_ERROR(
    cudaStreamSynchronize(stream),
    "Failed to synchronize CUDA stream in updateAllMemories");
  for (uint32_t idx = 0; idx < num_objects; idx++) {
    std::string object_id = output_mask_idx_to_obj_id_[idx];
    auto object = objects_[object_id];
    if (memory_timestamp < object->getMemoryUpdateTimestamp() ||
      memory_timestamp < object->getObjectUpdateTimestamp())
    {
      RCLCPP_WARN(
        node_->get_logger(),
        "Skipping object id: %s because memory timestamp is older than the object's "
        "update timestamp",
        object_id.c_str());
      continue;
    }

    // Don't update object if memory timestamp is older than the object's update timestamp
    auto object_score = object_scores[idx];
    // Don't update object if score is low
    if (object_score < 0) {
      continue;
    }

    const float * src_mask = mask_memories + (idx * kMaskMemPerIdxElements);
    const float * src_pos = mask_pos_enc + (idx * kMaskMemPerIdxElements);
    const float * src_ptr = obj_ptr_memories + (idx * kObjPtrMemPerIdxElements);

    // Check if this object is new
    bool is_new_object = false;
    if (object->getPromptType() == SAM2PromptType::BBOX) {
      auto it = std::find(new_bbox_ids_.begin(), new_bbox_ids_.end(), object_id);
      if (it != new_bbox_ids_.end()) {
        is_new_object = true;
        new_bbox_ids_.erase(it);
        RCLCPP_INFO(
          node_->get_logger(), "Removing object id: %s from new_bbox_ids_", object_id.c_str());
      }
    } else {
      auto it = std::find(new_point_ids_.begin(), new_point_ids_.end(), object_id);
      if (it != new_point_ids_.end()) {
        is_new_object = true;
        new_point_ids_.erase(it);
        RCLCPP_INFO(
          node_->get_logger(), "Removing object id: %s from new_point_ids_",
          object_id.c_str());
      }
    }
    // If the object is not new, and the memory timestamp is older than the object's
    // last frame with prompt timestamp, skip the update
    if (!is_new_object && memory_timestamp < object->getLastFrameWithPromptTimestamp()) {
      RCLCPP_DEBUG(
        node_->get_logger(),
        "Skipping object id: %s because Object has already received memory "
        "corresponding to a conditioned frame",
        object_id.c_str());
      continue;
    }
    object->updateTimestamp(TimestampType::MEMORY_UPDATE, memory_timestamp);
    // Update memory
    object->updateMemory(src_mask, src_pos, src_ptr, stream, is_new_object);
  }
  return;
}

// returns the object ids which weren't added
std::vector<std::string> SAM2StateManager::addObjects(
  const std::vector<std::string> & object_ids_bbox,
  const std::vector<std::string> & object_ids_points,
  const std::vector<BBox> & bbox_coords,
  const std::vector<std::vector<float>> & points,
  const std::vector<std::vector<int>> & labels,
  int64_t timestamp, cudaStream_t stream)
{
  std::vector<std::string> not_added_object_ids;
  for (uint32_t idx = 0; idx < object_ids_bbox.size(); idx++) {
    auto object_id = object_ids_bbox[idx];
    auto bbox = bbox_coords[idx];
    auto it = std::find(object_ids_points.begin(), object_ids_points.end(), object_id);
    if (it != object_ids_points.end()) {
      not_added_object_ids.push_back(object_id);
      continue;
    }
    auto result = addObject(object_id, SAM2PromptType::BBOX, timestamp, stream, bbox);
    if (!result) {
      not_added_object_ids.push_back(object_id);
    }
  }
  for (uint32_t idx = 0; idx < object_ids_points.size(); idx++) {
    auto object_id = object_ids_points[idx];
    auto point = points[idx];
    auto label = labels[idx];
    auto result = addObject(
      object_id, SAM2PromptType::POINTS, timestamp,
      stream, std::nullopt, point, label);
    if (!result) {
      not_added_object_ids.push_back(object_id);
    }
  }
  return not_added_object_ids;
}

void SAM2StateManager::getObjectIdsToOutputMaskIdx(
  std::vector<std::string> & object_ids, std::vector<int32_t> & output_mask_idx)
{
  for (const auto & obj_id : output_mask_idx_to_obj_id_) {
    object_ids.push_back(obj_id);
    output_mask_idx.push_back(obj_ids_to_output_mask_idx_.at(obj_id));
  }
}

std::vector<std::string> SAM2StateManager::getAllObjectIds()
{
  std::vector<std::string> object_ids;
  for (const auto & obj_id : objects_) {
    object_ids.push_back(obj_id.first);
  }
  return object_ids;
}
}  // namespace segment_anything2
}  // namespace isaac_ros
}  // namespace nvidia
