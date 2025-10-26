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

#include "isaac_ros_segment_anything2/segment_anything2_data_encoder_node.hpp"

#include "isaac_ros_nitros/types/nitros_type_base.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "vision_msgs/msg/bounding_box2_d.hpp"
#include "vision_msgs/msg/point2_d.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything2
{
namespace
{
constexpr char kDefaultQoS[] = "DEFAULT";

nvidia::isaac_ros::nitros::NitrosTensorShape getImageShape()
{
  return {1, 3, 1024, 1024};
}
nvidia::isaac_ros::nitros::NitrosTensorShape getMaskMemoryTensorShape(int32_t batch_size)
{
  return {batch_size, 4, 64, 64, 64};
}
nvidia::isaac_ros::nitros::NitrosTensorShape getObjPtrMemoryTensorShape(int32_t batch_size)
{
  return {batch_size, 2, 256};
}
nvidia::isaac_ros::nitros::NitrosTensorShape getBboxCoordsTensorShape(int32_t num_bbox_objects)
{
  return {num_bbox_objects, 4};
}
nvidia::isaac_ros::nitros::NitrosTensorShape getPointCoordsTensorShape(
  int32_t num_point_objects)
{
  return {num_point_objects, SAM2StateManager::kMaxPointsPerObject, 2};
}
nvidia::isaac_ros::nitros::NitrosTensorShape getPointLabelsTensorShape(
  int32_t num_point_objects)
{
  return {num_point_objects, SAM2StateManager::kMaxPointsPerObject};
}
nvidia::isaac_ros::nitros::NitrosTensorShape getPermutationTensorShape(int32_t batch_size)
{
  return {batch_size};
}
nvidia::isaac_ros::nitros::NitrosTensorShape getOriginalSizeTensorShape()
{
  return {2};
}

BBox getBboxCoords(const vision_msgs::msg::BoundingBox2D & bbox)
{
  float x_min = bbox.center.position.x - (bbox.size_x / 2);
  float y_min = bbox.center.position.y - (bbox.size_y / 2);
  float x_max = bbox.center.position.x + (bbox.size_x / 2);
  float y_max = bbox.center.position.y + (bbox.size_y / 2);
  return BBox(x_min, y_min, x_max, y_max);
}

}  // namespace


SegmentAnything2DataEncoderNode::SegmentAnything2DataEncoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("segment_anything2_data_encoder_node", options),
  image_qos_(::isaac_ros::common::AddQosParameter(*this, kDefaultQoS, "image_qos")),
  memory_qos_(::isaac_ros::common::AddQosParameter(*this, kDefaultQoS, "memory_qos")),
  encoded_data_qos_(::isaac_ros::common::AddQosParameter(*this, kDefaultQoS, "encoded_data_qos"))
{
  // Initialize parameters
  max_num_objects_ = declare_parameter<int32_t>("max_num_objects", 10);
  orig_img_dims_param_ = declare_parameter<std::vector<int64_t>>("orig_img_dims", {480, 640});
  if (orig_img_dims_param_.size() != 2) {
    throw std::runtime_error("orig_img_dims must be a vector of size 2");
  }
  // Initialize publisher for encoded data
  encoded_data_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosTensorList>>(
    this, "encoded_data",
    nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
    nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig(),
    encoded_data_qos_
        );

  // Initialize subscribers
  image_sub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
    this, "image",
    nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
    std::bind(&SegmentAnything2DataEncoderNode::ImageCallback, this, std::placeholders::_1),
    nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig(),
    image_qos_
        );

  memory_sub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
    this, "memory",
    nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
    std::bind(&SegmentAnything2DataEncoderNode::MemoryCallback, this, std::placeholders::_1),
    nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig(),
    memory_qos_
        );

  // Initialize service for adding objects
  add_objects_srv_ = create_service<isaac_ros_segment_anything2_interfaces::srv::AddObjects>(
    "add_objects",
    std::bind(
      &SegmentAnything2DataEncoderNode::AddObjectsCallback, this,
      std::placeholders::_1, std::placeholders::_2));

  // Initialize service for removing objects
  remove_object_srv_ = create_service<isaac_ros_segment_anything2_interfaces::srv::RemoveObject>(
    "remove_object",
    std::bind(
      &SegmentAnything2DataEncoderNode::RemoveObjectCallback, this,
      std::placeholders::_1, std::placeholders::_2));

  // Initialize state manager
  sam2_state_manager_ = std::make_unique<SAM2StateManager>(this);
  ::nvidia::isaac_ros::common::initNamedCudaStream(
    stream_, "isaac_ros_segment_anything2_data_encoder_node");
  // Allocate original size buffer
  std::vector<int32_t> orig_img_dims = std::vector<int32_t>(
    orig_img_dims_param_.begin(),
    orig_img_dims_param_.end());
  RCLCPP_INFO(get_logger(), "orig_img_dims: %d, %d", orig_img_dims[0], orig_img_dims[1]);
  CHECK_CUDA_ERROR(
    cudaMallocAsync(&original_size_buffer_, 2 * sizeof(int32_t), stream_),
    "Failed to allocate original size buffer");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      original_size_buffer_, orig_img_dims.data(), 2 * sizeof(int32_t),
      cudaMemcpyHostToDevice, stream_),
    "Failed to copy original size buffer");
  RCLCPP_INFO(get_logger(), "SegmentAnything2DataEncoderNode initialized");
}

SegmentAnything2DataEncoderNode::~SegmentAnything2DataEncoderNode()
{
  RCLCPP_INFO(get_logger(), "SegmentAnything2DataEncoderNode destroyed");
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_), "Failed to destroy CUDA stream");
}

void SegmentAnything2DataEncoderNode::ImageCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & view)
{
  RCLCPP_DEBUG(get_logger(), "Received image tensor");
  int num_objects = sam2_state_manager_->getNumberOfObjects();
  if (num_objects == 0) {
    RCLCPP_DEBUG(get_logger(), "No objects found in the state manager!");
    return;
  }
  auto input_tensor = view.GetNamedTensor("input_tensor");

  // Get timestamp from the message
  int64_t timestamp = 0;
  try {
    timestamp = view.GetTimestampSeconds() * 1000000000LL + view.GetTimestampNanoseconds();
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to get timestamp from message");
  }
  void * image_buffer;
  CHECK_CUDA_ERROR(
    cudaMallocAsync(&image_buffer, input_tensor.GetTensorSize(), stream_),
    "Failed to allocate image buffer");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      image_buffer, input_tensor.GetBuffer(),
      input_tensor.GetTensorSize(), cudaMemcpyDeviceToDevice, stream_),
    "Failed to copy image buffer");

  int32_t * original_size_buffer;
  CHECK_CUDA_ERROR(
    cudaMallocAsync(&original_size_buffer, 2 * sizeof(int32_t), stream_),
    "Failed to allocate original size buffer");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      original_size_buffer, original_size_buffer_, 2 * sizeof(int32_t),
      cudaMemcpyDeviceToDevice, stream_),
    "Failed to copy original size buffer");

  SAM2BufferData buffer_data = sam2_state_manager_->getBuffers(stream_, timestamp);

  // Add a second check, object might have been removed between the two checks
  if (buffer_data.batch_size == 0) {
    RCLCPP_WARN(get_logger(), "No objects found in the state manager!");
    return;
  }

  std_msgs::msg::Header header;
  header.stamp.sec = view.GetTimestampSeconds();
  header.stamp.nanosec = view.GetTimestampNanoseconds();
  header.frame_id = view.GetFrameId();
  auto image_tensor = nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithShape(getImageShape())
    .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
    .WithData(image_buffer)
    .Build();
  auto mask_memory_tensor = nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithShape(getMaskMemoryTensorShape(buffer_data.batch_size))
    .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
    .WithData(buffer_data.mask_mem)
    .Build();
  auto obj_ptr_memory_tensor = nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithShape(getObjPtrMemoryTensorShape(buffer_data.batch_size))
    .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
    .WithData(buffer_data.obj_ptr_mem)
    .Build();
  auto bbox_coords_tensor = nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithShape(getBboxCoordsTensorShape(buffer_data.num_bboxes))
    .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
    .WithData(buffer_data.bbox_coords)
    .Build();
  auto point_coords_tensor = nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithShape(getPointCoordsTensorShape(buffer_data.num_points))
    .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
    .WithData(buffer_data.point_coords)
    .Build();
  auto point_labels_tensor = nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithShape(getPointLabelsTensorShape(buffer_data.num_points))
    .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kInt32)
    .WithData(buffer_data.point_labels)
    .Build();
  auto permutation_tensor = nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithShape(getPermutationTensorShape(buffer_data.batch_size))
    .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kInt64)
    .WithData(buffer_data.permutation)
    .Build();
  auto original_size_tensor = nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithShape(getOriginalSizeTensorShape())
    .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kInt32)
    .WithData(original_size_buffer)
    .Build();
  nvidia::isaac_ros::nitros::NitrosTensorList tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor("image", image_tensor)
    .AddTensor("original_size", original_size_tensor)
    .AddTensor("mask_memory", mask_memory_tensor)
    .AddTensor("obj_ptr_memory", obj_ptr_memory_tensor)
    .AddTensor("bbox_coords", bbox_coords_tensor)
    .AddTensor("point_coords", point_coords_tensor)
    .AddTensor("point_labels", point_labels_tensor)
    .AddTensor("permutation", permutation_tensor)
    .Build();
  CHECK_CUDA_ERROR(
    cudaStreamSynchronize(stream_), "Failed to synchronize CUDA stream in ImageCallback");
  encoded_data_pub_->publish(tensor_list);
}

void SegmentAnything2DataEncoderNode::MemoryCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & view)
{
  auto object_score_logits = view.GetNamedTensor("object_score_logits");
  auto maskmem_features = view.GetNamedTensor("maskmem_features");
  auto maskmem_pos_enc = view.GetNamedTensor("maskmem_pos_enc");
  auto obj_ptr_features = view.GetNamedTensor("obj_ptr_features");
  // Get timestamp from the message
  int64_t batch_size = object_score_logits.GetShape().shape().dimension(0);
  int64_t timestamp = 0;
  try {
    timestamp = view.GetTimestampSeconds() * 1000000000LL + view.GetTimestampNanoseconds();
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to get timestamp from message");
  }
  sam2_state_manager_->updateAllMemories(
    reinterpret_cast<const float *>(maskmem_features.GetBuffer()),
    reinterpret_cast<const float *>(maskmem_pos_enc.GetBuffer()),
    reinterpret_cast<const float *>(obj_ptr_features.GetBuffer()),
    reinterpret_cast<const float *>(object_score_logits.GetBuffer()),
    stream_,
    batch_size,
    timestamp
  );
}

void SegmentAnything2DataEncoderNode::AddObjectsCallback(
  const std::shared_ptr<isaac_ros_segment_anything2_interfaces::srv::AddObjects::Request> request,
  std::shared_ptr<isaac_ros_segment_anything2_interfaces::srv::AddObjects::Response> response)
{
  RCLCPP_INFO(get_logger(), "Received add_objects request");
  if (request->bbox_coords.size() != request->bbox_object_ids.size()) {
    response->success = false;
    response->message = "Number of bbox_object_ids doesn't match number of bbox_coords";
    return;
  }

  if (request->point_coords.size() != request->point_object_ids.size()) {
    response->success = false;
    response->message = "Number of point_object_ids doesn't match number of point_coords";
    return;
  }

  if (request->point_coords.size() != request->point_labels.size()) {
    response->success = false;
    response->message = "Number of point labels doesn't match number of points";
    return;
  }
  std::vector<std::string> all_object_ids = sam2_state_manager_->getAllObjectIds();

  // Check if all bbox_object_ids are unique
  for (int i = 0; i < request->bbox_object_ids.size(); i++) {
    auto bbox_id_it = std::find(
      request->bbox_object_ids.begin() + i + 1,
      request->bbox_object_ids.end(),
      request->bbox_object_ids[i]);
    auto point_id_it = std::find(
      request->point_object_ids.begin(),
      request->point_object_ids.end(),
      request->bbox_object_ids[i]);
    auto all_object_id_it = std::find(
      all_object_ids.begin(),
      all_object_ids.end(),
      request->bbox_object_ids[i]);
    if (bbox_id_it != request->bbox_object_ids.end() ||
      point_id_it != request->point_object_ids.end())
    {
      RCLCPP_WARN(
        get_logger(),
        "duplicate bbox_object_id: %s exists in point_object_ids or bbox_object_ids",
        request->bbox_object_ids[i].c_str());
      response->success = false;
      response->message = "duplicate bbox_object_id: " +
        request->bbox_object_ids[i] +
        " exists in point_object_ids or bbox_object_ids";
      return;
    }
    if (all_object_id_it != all_object_ids.end()) {
      RCLCPP_WARN(
        get_logger(),
        "duplicate bbox_object_id: %s already exists",
        request->bbox_object_ids[i].c_str());
      response->success = false;
      response->message = "duplicate bbox_object_id: " +
        request->bbox_object_ids[i] +
        " already exists";
      return;
    }
  }

  // Merge point_coords and point_labels for each object
  std::vector<std::vector<float>> merged_point_coords;
  std::vector<std::vector<int>> merged_point_labels;
  std::vector<std::string> unique_point_ids;
  for (int i = 0; i < request->point_object_ids.size(); i++) {
    auto point_id_it = std::find(
      unique_point_ids.begin(),
      unique_point_ids.end(), request->point_object_ids[i]);
    std::string point_id = request->point_object_ids[i];
    vision_msgs::msg::Point2D point_coord = request->point_coords[i];
    float point_x = point_coord.x;
    float point_y = point_coord.y;
    int point_label = request->point_labels[i];
    if (point_id_it == unique_point_ids.end()) {
      // Check if the point_object_id is already in the all_object_ids vector
      auto all_object_id_it = std::find(all_object_ids.begin(), all_object_ids.end(), point_id);
      if (all_object_id_it != all_object_ids.end()) {
        RCLCPP_WARN(
          get_logger(),
          "duplicate point_object_id: %s already exists", point_id.c_str());
        response->success = false;
        response->message = "duplicate point_object_id: " + point_id + " already exists.";
        return;
      }
      unique_point_ids.push_back(point_id);
      merged_point_coords.push_back(std::vector<float>{point_x, point_y});
      merged_point_labels.push_back(std::vector<int>{point_label});
    } else {
      int idx = std::distance(unique_point_ids.begin(), point_id_it);
      merged_point_coords[idx].push_back(point_x);
      merged_point_coords[idx].push_back(point_y);
      merged_point_labels[idx].push_back(point_label);
      if (merged_point_labels[idx].size() > SAM2StateManager::kMaxPointsPerObject) {
        RCLCPP_WARN(
          get_logger(),
          "Number of points for object %s exceeds the maximum limit of %d", point_id.c_str(),
          SAM2StateManager::kMaxPointsPerObject);
        response->success = false;
        response->message = "Number of points for object " +
          point_id + " exceeds the maximum limit of " + std::to_string(
          SAM2StateManager::kMaxPointsPerObject);
        return;
      }
    }
  }

  if (request->bbox_object_ids.size() +
    unique_point_ids.size() + all_object_ids.size() > max_num_objects_)
  {
    RCLCPP_WARN(
      get_logger(),
      "Number of objects exceeds the maximum limit of %d", max_num_objects_);
    response->success = false;
    response->message = "Number of objects exceeds the maximum limit of " +
      std::to_string(max_num_objects_);
    return;
  }

  int64_t timestamp = static_cast<int64_t>(request->request_header.stamp.sec) * 1000000000LL +
    static_cast<int64_t>(request->request_header.stamp.nanosec);

  std::vector<BBox> bbox_coords;
  for (int i = 0; i < request->bbox_object_ids.size(); i++) {
    bbox_coords.push_back(getBboxCoords(request->bbox_coords[i]));
  }

  std::vector<std::string> not_added_object_ids = sam2_state_manager_->addObjects(
    request->bbox_object_ids, unique_point_ids, bbox_coords, merged_point_coords,
    merged_point_labels, timestamp, stream_);
  std::vector<std::string> object_ids;
  std::vector<int32_t> output_mask_idx;
  sam2_state_manager_->getObjectIdsToOutputMaskIdx(object_ids, output_mask_idx);
  for (int i = 0; i < object_ids.size(); i++) {
    response->object_ids.push_back(object_ids[i]);
    response->object_indices.push_back(output_mask_idx[i]);
  }
  // Since we are checking for duplicates already, all the objects should be added
  if (!not_added_object_ids.empty()) {
    throw std::runtime_error("Failed to add all objects");
  }
  response->success = true;
  response->message = "Success";  // Set based on actual processing result
}

void SegmentAnything2DataEncoderNode::RemoveObjectCallback(
  const std::shared_ptr<isaac_ros_segment_anything2_interfaces::srv::RemoveObject::Request> request,
  std::shared_ptr<isaac_ros_segment_anything2_interfaces::srv::RemoveObject::Response> response)
{
  RCLCPP_INFO(get_logger(), "Received remove_object request");
  std::string object_id = request->object_id;
  int64_t timestamp = static_cast<int64_t>(request->request_header.stamp.sec) * 1000000000LL +
    static_cast<int64_t>(request->request_header.stamp.nanosec);
  bool result = sam2_state_manager_->removeObject(object_id, timestamp);
  if (!result) {
    RCLCPP_WARN(get_logger(), "Failed to remove object with id: %s", object_id.c_str());
    response->success = false;
    response->message = "Failed to remove object";
    return;
  }
  std::vector<std::string> object_ids;
  std::vector<int32_t> output_mask_idx;
  sam2_state_manager_->getObjectIdsToOutputMaskIdx(object_ids, output_mask_idx);
  for (int i = 0; i < object_ids.size(); i++) {
    response->object_ids.push_back(object_ids[i]);
    response->object_indices.push_back(output_mask_idx[i]);
  }
  response->success = true;
  response->message = "Success";
}

}  // namespace segment_anything2
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with class_loader
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::segment_anything2::SegmentAnything2DataEncoderNode)
