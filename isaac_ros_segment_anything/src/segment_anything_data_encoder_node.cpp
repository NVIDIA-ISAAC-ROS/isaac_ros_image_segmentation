// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_segment_anything/segment_anything_data_encoder_node.hpp"

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_shape.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_data_type.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

namespace
{

bool IsSupportedInputPromptType(const std::string & prompt_type)
{
  return prompt_type == "bbox" || prompt_type == "point";
}

}  // namespace

SegmentAnythingDataEncoderNode::SegmentAnythingDataEncoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("segment_anything_data_encoder", options),
  max_batch_size_(declare_parameter<int32_t>("max_batch_size", 20)),
  prompt_input_type_(declare_parameter<std::string>("prompt_input_type", "bbox")),
  has_input_mask_(declare_parameter<bool>("has_input_mask", false)),
  orig_img_dims_(declare_parameter<std::vector<int64_t>>("orig_img_dims", {632, 1200}))
{
  if (!IsSupportedInputPromptType(prompt_input_type_)) {
    RCLCPP_ERROR(
      get_logger(),
      "Received invalid input prompt type: %s!",
      prompt_input_type_.c_str());
    throw std::invalid_argument("Received invalid input prompt type." + prompt_input_type_);
  }

  is_bbox_prompt_ = (prompt_input_type_ == "bbox");

  // Compute resized dimensions (matching SAM preprocessing)
  uint32_t orig_width = orig_img_dims_[1];
  uint32_t orig_height = orig_img_dims_[0];
  if (orig_width > orig_height) {
    resized_width_ = kImageWidth;
    resized_height_ = static_cast<uint16_t>(
      (static_cast<float>(resized_width_) / orig_width) * orig_height);
  } else {
    resized_height_ = kImageHeight;
    resized_width_ = static_cast<uint16_t>(
      (static_cast<float>(resized_height_) / orig_height) * orig_width);
  }

  // Create CUDA stream
  const cudaError_t stream_err = cudaStreamCreate(&cuda_stream_);
  if (stream_err != cudaSuccess) {
    cuda_stream_ = nullptr;
    RCLCPP_ERROR(
      get_logger(),
      "Failed to create CUDA stream: %s",
      cudaGetErrorString(stream_err));
    throw std::runtime_error(
            std::string("cudaStreamCreate failed: ") + cudaGetErrorString(stream_err));
  }

  // Initialize publisher
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  output_pub_ = create_publisher<NitrosTensorList>("tensor", rclcpp::QoS(1), pub_options);

  // Initialize synchronizer before subscribing
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  exact_sync_ = std::make_shared<ExactSync>(
    ExactPolicy(10), prompt_sub_, image_sub_, mask_sub_);
  exact_sync_->registerCallback(
    std::bind(
      &SegmentAnythingDataEncoderNode::SyncCallback, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  // Subscribe after registering the callback
  prompt_sub_.subscribe(this, "prompts", rclcpp::QoS(1).get_rmw_qos_profile(), sub_options);
  image_sub_.subscribe(this, "tensor_pub", rclcpp::QoS(1).get_rmw_qos_profile(), sub_options);
  mask_sub_.subscribe(this, "mask", rclcpp::QoS(1).get_rmw_qos_profile(), sub_options);
}

SegmentAnythingDataEncoderNode::~SegmentAnythingDataEncoderNode()
{
  if (cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
  }
}

void SegmentAnythingDataEncoderNode::SyncCallback(
  const Detection2DArray::ConstSharedPtr & prompts,
  const NitrosTensorList::ConstSharedPtr & image_tensor,
  const NitrosTensorList::ConstSharedPtr & mask_tensor)
{
  const auto & detections = prompts->detections;

  // No detections means no inference on this frame
  if (detections.empty()) {
    RCLCPP_INFO(get_logger(), "No input prompt found. No inference would run on this frame.");
    return;
  }

  // Determine batch size and number of points
  uint32_t batch_size = 1;
  uint32_t num_points = kNumPointsPerBbox;
  if (is_bbox_prompt_) {
    batch_size = static_cast<uint32_t>(
      max_batch_size_ < static_cast<int32_t>(detections.size()) ?
      max_batch_size_ : detections.size());
  } else {
    // Point prompt: +1 for padding point
    num_points = static_cast<uint32_t>(detections.size()) + 1;
  }

  // Convert detections to SAM prompt format
  std::vector<float> prompt_vec;
  std::vector<float> label_vec;
  DetectionToSAMPrompt(detections, prompt_vec, label_vec);

  // Each tensor: cudaMallocAsync, wrap via from_external (ownership transfers,
  // default cudaFree deleter), write through the returned WriteHandle. The handle
  // dropping records a CUDA event so consumers sync via get_read_handle.
  auto alloc_h2d_tensor =
    [&](const std::string & name, const void * src, size_t bytes,
    const nitros::NitrosTensorShape & shape,
    nitros::NitrosDataType dtype, nitros::NitrosTensor & out) -> bool {
      void * gpu_ptr = nullptr;
      const cudaError_t err = cudaMallocAsync(&gpu_ptr, bytes, cuda_stream_);
      if (err != cudaSuccess) {
        RCLCPP_ERROR(
          get_logger(), "cudaMallocAsync failed for tensor '%s' (%zu bytes): %s",
          name.c_str(), bytes, cudaGetErrorString(err));
        return false;
      }
      auto wh = out.from_external(name, gpu_ptr, bytes, shape, dtype, cuda_stream_);
      const cudaError_t copy_err = cudaMemcpyAsync(
        wh.get_ptr(), src, bytes, cudaMemcpyHostToDevice, cuda_stream_);
      if (copy_err != cudaSuccess) {
        RCLCPP_ERROR(
          get_logger(), "cudaMemcpyAsync H2D failed for tensor '%s' (%zu bytes): %s",
          name.c_str(), bytes, cudaGetErrorString(copy_err));
        return false;
      }
      return true;
    };

  // Prompt tensors (host->device copies).
  const uint32_t points_buffer_size = batch_size * num_points * 2 * sizeof(float);
  nitros::NitrosTensor points_tensor;
  if (!alloc_h2d_tensor(
      "points", prompt_vec.data(), points_buffer_size,
      nitros::NitrosTensorShape(
        {static_cast<int32_t>(batch_size), static_cast<int32_t>(num_points), 2}),
      nitros::NitrosDataType::kFloat32, points_tensor))
  {
    return;
  }

  const uint32_t labels_buffer_size = batch_size * num_points * sizeof(float);
  nitros::NitrosTensor labels_tensor;
  if (!alloc_h2d_tensor(
      "labels", label_vec.data(), labels_buffer_size,
      nitros::NitrosTensorShape(
        {static_cast<int32_t>(batch_size), static_cast<int32_t>(num_points)}),
      nitros::NitrosDataType::kFloat32, labels_tensor))
  {
    return;
  }

  const std::vector<float> has_mask_data = {has_input_mask_ ? 1.0f : 0.0f};
  nitros::NitrosTensor has_mask_tensor;
  if (!alloc_h2d_tensor(
      "has_input_mask", has_mask_data.data(), sizeof(float),
      nitros::NitrosTensorShape({1}),
      nitros::NitrosDataType::kFloat32, has_mask_tensor))
  {
    return;
  }

  const std::vector<float> img_size_data = {
    static_cast<float>(orig_img_dims_[0]),
    static_cast<float>(orig_img_dims_[1])};
  nitros::NitrosTensor img_size_tensor;
  if (!alloc_h2d_tensor(
      "orig_img_dims", img_size_data.data(), 2 * sizeof(float),
      nitros::NitrosTensorShape({2}),
      nitros::NitrosDataType::kFloat32, img_size_tensor))
  {
    return;
  }

  // Forward tensors from image and mask inputs (device->device copies).
  std::vector<std::pair<std::string, nitros::NitrosTensor>> forwarded_tensors;

  auto forward_tensors_from_msg =
    [&](const nitros::NitrosTensorList & msg) -> bool {
      for (const auto & input : msg.get_tensors()) {
        const size_t tensor_bytes = input.element_count() * input.bytes_per_element();
        void * gpu_copy = nullptr;
        const cudaError_t err = cudaMallocAsync(&gpu_copy, tensor_bytes, cuda_stream_);
        if (err != cudaSuccess) {
          RCLCPP_ERROR(
            get_logger(),
            "cudaMallocAsync failed forwarding tensor '%s' (%zu bytes): %s",
            input.get_name().c_str(), tensor_bytes, cudaGetErrorString(err));
          return false;
        }
        nitros::NitrosTensor copy;
        auto wh = copy.from_external(
          input.get_name(), gpu_copy, tensor_bytes,
          input.shape(), input.data_type(), cuda_stream_);
        // Consumer passes cuda_stream_ so this stream waits on the producer's event.
        auto input_read_handle = input.get_read_handle(cuda_stream_);
        const cudaError_t copy_err = cudaMemcpyAsync(
          wh.get_ptr(), input_read_handle.get_ptr(), tensor_bytes,
          cudaMemcpyDeviceToDevice, cuda_stream_);
        if (copy_err != cudaSuccess) {
          RCLCPP_ERROR(
            get_logger(),
            "cudaMemcpyAsync D2D failed forwarding tensor '%s' (%zu bytes): %s",
            input.get_name().c_str(), tensor_bytes, cudaGetErrorString(copy_err));
          return false;
        }
        forwarded_tensors.emplace_back(input.get_name(), std::move(copy));
      }
      return true;
    };

  if (!forward_tensors_from_msg(*image_tensor)) {
    return;
  }
  if (!forward_tensors_from_msg(*mask_tensor)) {
    return;
  }

  // Sync is still required: the prompt tensor H2D copies source from local
  // std::vectors that go out of scope when this callback returns.
  const cudaError_t sync_err = cudaStreamSynchronize(cuda_stream_);
  if (sync_err != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(), "cudaStreamSynchronize failed before publish: %s",
      cudaGetErrorString(sync_err));
    return;
  }

  std_msgs::msg::Header header;
  header.stamp = prompts->header.stamp;
  header.frame_id = prompts->header.frame_id;

  auto builder = nitros::NitrosTensorListBuilder().WithHeader(header);
  builder.AddTensor("points", std::move(points_tensor));
  builder.AddTensor("labels", std::move(labels_tensor));
  builder.AddTensor("orig_img_dims", std::move(img_size_tensor));
  builder.AddTensor("has_input_mask", std::move(has_mask_tensor));
  for (auto & ft : forwarded_tensors) {
    builder.AddTensor(ft.first, std::move(ft.second));
  }
  output_pub_->publish(builder.Build());
}

void SegmentAnythingDataEncoderNode::DetectionToSAMPrompt(
  const std::vector<vision_msgs::msg::Detection2D> & detections,
  std::vector<float> & prompt_vec,
  std::vector<float> & label_vec)
{
  uint32_t orig_width = orig_img_dims_[1];
  uint32_t orig_height = orig_img_dims_[0];
  float width_scale = static_cast<float>(resized_width_) / orig_width;
  float height_scale = static_cast<float>(resized_height_) / orig_height;

  for (const auto & det : detections) {
    double center_x = det.bbox.center.position.x;
    double center_y = det.bbox.center.position.y;
    double size_x = det.bbox.size_x;
    double size_y = det.bbox.size_y;

    if (is_bbox_prompt_) {
      float top_left_x = static_cast<float>((center_x - size_x / 2.0) * width_scale);
      float top_left_y = static_cast<float>((center_y - size_y / 2.0) * height_scale);
      float bottom_right_x = static_cast<float>((center_x + size_x / 2.0) * width_scale);
      float bottom_right_y = static_cast<float>((center_y + size_y / 2.0) * height_scale);
      prompt_vec.push_back(top_left_x);
      prompt_vec.push_back(top_left_y);
      prompt_vec.push_back(bottom_right_x);
      prompt_vec.push_back(bottom_right_y);
      label_vec.push_back(2.0f);
      label_vec.push_back(3.0f);
    } else {
      prompt_vec.push_back(static_cast<float>(center_x * width_scale));
      prompt_vec.push_back(static_cast<float>(center_y * height_scale));
      label_vec.push_back(1.0f);
    }
  }

  // Add padding point for point prompts
  if (!is_bbox_prompt_) {
    prompt_vec.push_back(0.0f);
    prompt_vec.push_back(0.0f);
    label_vec.push_back(-1.0f);
  }
}

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode)
