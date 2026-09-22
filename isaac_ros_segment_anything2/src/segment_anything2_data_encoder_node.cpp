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

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"
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

using Tensor = tensor_msgs::msg::ExperimentalTensor;
using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

// DLPack dtypes used by this node: {dtype_code, dtype_bits} with dtype_lanes = 1
// (see ExperimentalTensor.msg).
struct DLDtype
{
  uint8_t code;
  uint8_t bits;
};
constexpr DLDtype kDtypeInt32 = {0, 32};
constexpr DLDtype kDtypeInt64 = {0, 64};
constexpr DLDtype kDtypeFloat32 = {2, 32};

// Tensor shapes as row-major dimension vectors.
std::vector<int64_t> getImageShape()
{
  return {1, 3, 1024, 1024};
}
std::vector<int64_t> getMaskMemoryTensorShape(int32_t batch_size)
{
  return {batch_size, 4, 64, 64, 64};
}
std::vector<int64_t> getObjPtrMemoryTensorShape(int32_t batch_size)
{
  return {batch_size, 2, 256};
}
std::vector<int64_t> getBboxCoordsTensorShape(int32_t num_bbox_objects)
{
  return {num_bbox_objects, 4};
}
std::vector<int64_t> getPointCoordsTensorShape(int32_t num_point_objects)
{
  return {num_point_objects,
    static_cast<int64_t>(SAM2StateManager::kMaxPointsPerObject), 2};
}
std::vector<int64_t> getPointLabelsTensorShape(int32_t num_point_objects)
{
  return {num_point_objects,
    static_cast<int64_t>(SAM2StateManager::kMaxPointsPerObject)};
}
std::vector<int64_t> getPermutationTensorShape(int32_t batch_size)
{
  return {batch_size};
}
std::vector<int64_t> getOriginalSizeTensorShape()
{
  return {2};
}

size_t ElementCount(const std::vector<int64_t> & dims)
{
  size_t n = 1;
  for (auto d : dims) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

// Look up a tensor by name within a TensorList.
// Tensor names live in the TensorList-level names array, parallel to tensors.
const Tensor * findTensor(const TensorList & msg, const std::string & name)
{
  for (size_t i = 0; i < msg.names.size() && i < msg.tensors.size(); ++i) {
    if (msg.names[i] == name) {
      return &msg.tensors[i];
    }
  }
  return nullptr;
}

// Build a Tensor by allocating a cuda_buffer-backed output buffer and D2D-copying
// from a device source pointer. The WriteHandle leaving scope records a CUDA event
// on the stream so consumers synchronize via from_input_buffer.
Tensor makeDeviceTensor(
  const void * src_dev,
  const std::vector<int64_t> & dims, DLDtype dtype, size_t elem_size,
  cudaStream_t stream)
{
  Tensor t;
  t.dtype_code = dtype.code;
  t.dtype_bits = dtype.bits;
  t.dtype_lanes = 1;
  t.shape = dims;
  // strides left empty: contiguous row-major per DLPack convention
  t.byte_offset = 0;
  const size_t bytes = ElementCount(dims) * elem_size;
  t.data = cuda_buffer_backend::allocate_buffer(bytes);
  // A zero-element tensor (e.g. no bbox or no point objects this frame) is still
  // published with its shape; skip the handle/copy since from_output_buffer
  // rejects empty buffers.
  if (bytes > 0) {
    auto wh = cuda_buffer_backend::from_output_buffer(t.data, stream);
    cuda_buffer_backend::to_buffer(src_dev, bytes, wh, stream, cudaMemcpyDeviceToDevice);
  }
  return t;
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
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  // Initialize publisher for encoded data
  encoded_data_pub_ = create_publisher<TensorList>(
    "encoded_data", encoded_data_qos_, pub_options);

  // Initialize subscribers
  image_sub_ = create_subscription<TensorList>(
    "image", image_qos_,
    std::bind(&SegmentAnything2DataEncoderNode::ImageCallback, this, std::placeholders::_1),
    sub_options);

  memory_sub_ = create_subscription<TensorList>(
    "memory", memory_qos_,
    std::bind(&SegmentAnything2DataEncoderNode::MemoryCallback, this, std::placeholders::_1),
    sub_options);

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
  const TensorList::ConstSharedPtr & msg)
{
  RCLCPP_DEBUG(get_logger(), "Received image tensor");
  int num_objects = sam2_state_manager_->getNumberOfObjects();
  if (num_objects == 0) {
    RCLCPP_DEBUG(get_logger(), "No objects found in the state manager!");
    return;
  }
  const Tensor * input_tensor = findTensor(*msg, "input_tensor");
  if (!input_tensor) {
    throw std::runtime_error("Tensor with name 'input_tensor' not found");
  }

  int64_t timestamp = static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
    static_cast<int64_t>(msg->header.stamp.nanosec);

  // Image tensor: allocate a cuda_buffer-backed output and D2D-copy from the
  // subscribed input tensor's buffer. The read/write handles leaving scope each
  // record a CUDA event so consumers synchronize via from_input_buffer.
  const size_t image_bytes = input_tensor->data.size();
  const auto image_dims = getImageShape();
  Tensor image_tensor;
  image_tensor.dtype_code = kDtypeFloat32.code;
  image_tensor.dtype_bits = kDtypeFloat32.bits;
  image_tensor.dtype_lanes = 1;
  image_tensor.shape = image_dims;
  // strides left empty: contiguous row-major per DLPack convention. The whole
  // underlying buffer is copied, so the input's view offset carries over.
  image_tensor.byte_offset = input_tensor->byte_offset;
  image_tensor.data = cuda_buffer_backend::allocate_buffer(image_bytes);
  {
    auto image_write_handle = cuda_buffer_backend::from_output_buffer(image_tensor.data, stream_);
    auto input_read_handle = cuda_buffer_backend::from_input_buffer(input_tensor->data, stream_);
    cuda_buffer_backend::to_buffer(
      input_read_handle.get_ptr(), image_bytes, image_write_handle, stream_,
      cudaMemcpyDeviceToDevice);
  }

  // Original-size tensor: D2D-copy from the node's cached device buffer.
  Tensor original_size_tensor = makeDeviceTensor(
    original_size_buffer_, getOriginalSizeTensorShape(),
    kDtypeInt32, sizeof(int32_t), stream_);

  SAM2BufferData buffer_data = sam2_state_manager_->getBuffers(stream_, timestamp);

  // Add a second check, object might have been removed between the two checks
  if (buffer_data.batch_size == 0) {
    RCLCPP_WARN(get_logger(), "No objects found in the state manager!");
    return;
  }

  // getBuffers() allocates fresh buffers for this frame. Copy them into
  // cuda_buffer-backed tensors, then release the source buffers on the same stream.
  Tensor mask_memory_tensor = makeDeviceTensor(
    buffer_data.mask_mem, getMaskMemoryTensorShape(buffer_data.batch_size),
    kDtypeFloat32, sizeof(float), stream_);
  Tensor obj_ptr_memory_tensor = makeDeviceTensor(
    buffer_data.obj_ptr_mem, getObjPtrMemoryTensorShape(buffer_data.batch_size),
    kDtypeFloat32, sizeof(float), stream_);
  Tensor bbox_coords_tensor = makeDeviceTensor(
    buffer_data.bbox_coords, getBboxCoordsTensorShape(buffer_data.num_bboxes),
    kDtypeFloat32, sizeof(float), stream_);
  Tensor point_coords_tensor = makeDeviceTensor(
    buffer_data.point_coords, getPointCoordsTensorShape(buffer_data.num_points),
    kDtypeFloat32, sizeof(float), stream_);
  Tensor point_labels_tensor = makeDeviceTensor(
    buffer_data.point_labels, getPointLabelsTensorShape(buffer_data.num_points),
    kDtypeInt32, sizeof(int32_t), stream_);
  Tensor permutation_tensor = makeDeviceTensor(
    buffer_data.permutation, getPermutationTensorShape(buffer_data.batch_size),
    kDtypeInt64, sizeof(int64_t), stream_);

  CHECK_CUDA_ERROR(cudaFreeAsync(buffer_data.mask_mem, stream_), "Failed to free mask memory");
  CHECK_CUDA_ERROR(
    cudaFreeAsync(buffer_data.obj_ptr_mem, stream_), "Failed to free object pointer");
  CHECK_CUDA_ERROR(
    cudaFreeAsync(buffer_data.bbox_coords, stream_), "Failed to free bounding boxes");
  CHECK_CUDA_ERROR(
    cudaFreeAsync(buffer_data.point_coords, stream_), "Failed to free point coords");
  CHECK_CUDA_ERROR(
    cudaFreeAsync(buffer_data.point_labels, stream_), "Failed to free point labels");
  CHECK_CUDA_ERROR(cudaFreeAsync(buffer_data.permutation, stream_), "Failed to free permutation");

  // Tensor names live in the TensorList-level parallel names array; consumers
  // such as TensorRTNode look tensors up through it, so keep both arrays in
  // the same order.
  TensorList tensor_list;
  tensor_list.header = msg->header;
  tensor_list.names = {"image", "original_size", "mask_memory", "obj_ptr_memory",
    "bbox_coords", "point_coords", "point_labels", "permutation"};
  tensor_list.tensors.push_back(std::move(image_tensor));
  tensor_list.tensors.push_back(std::move(original_size_tensor));
  tensor_list.tensors.push_back(std::move(mask_memory_tensor));
  tensor_list.tensors.push_back(std::move(obj_ptr_memory_tensor));
  tensor_list.tensors.push_back(std::move(bbox_coords_tensor));
  tensor_list.tensors.push_back(std::move(point_coords_tensor));
  tensor_list.tensors.push_back(std::move(point_labels_tensor));
  tensor_list.tensors.push_back(std::move(permutation_tensor));
  encoded_data_pub_->publish(std::move(tensor_list));
}

void SegmentAnything2DataEncoderNode::MemoryCallback(
  const TensorList::ConstSharedPtr & msg)
{
  const Tensor * object_score_logits = findTensor(*msg, "object_score_logits");
  const Tensor * maskmem_features = findTensor(*msg, "maskmem_features");
  const Tensor * maskmem_pos_enc = findTensor(*msg, "maskmem_pos_enc");
  const Tensor * obj_ptr_features = findTensor(*msg, "obj_ptr_features");
  if (!object_score_logits || !maskmem_features || !maskmem_pos_enc || !obj_ptr_features) {
    throw std::runtime_error("Missing expected tensor in memory message");
  }
  int64_t batch_size = object_score_logits->shape[0];
  int64_t timestamp = static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
    static_cast<int64_t>(msg->header.stamp.nanosec);
  auto maskmem_features_handle =
    cuda_buffer_backend::from_input_buffer(maskmem_features->data, stream_);
  auto maskmem_pos_enc_handle =
    cuda_buffer_backend::from_input_buffer(maskmem_pos_enc->data, stream_);
  auto obj_ptr_features_handle =
    cuda_buffer_backend::from_input_buffer(obj_ptr_features->data, stream_);
  auto object_score_logits_handle =
    cuda_buffer_backend::from_input_buffer(object_score_logits->data, stream_);
  sam2_state_manager_->updateAllMemories(
    reinterpret_cast<const float *>(
      maskmem_features_handle.get_ptr() + maskmem_features->byte_offset),
    reinterpret_cast<const float *>(
      maskmem_pos_enc_handle.get_ptr() + maskmem_pos_enc->byte_offset),
    reinterpret_cast<const float *>(
      obj_ptr_features_handle.get_ptr() + obj_ptr_features->byte_offset),
    reinterpret_cast<const float *>(
      object_score_logits_handle.get_ptr() + object_score_logits->byte_offset),
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
