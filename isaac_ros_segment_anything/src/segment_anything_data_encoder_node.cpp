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

template<typename F>
class ScopeExit
{
public:
  explicit ScopeExit(F && f)
  : f_(std::forward<F>(f)) {}
  ~ScopeExit() {f_();}
  ScopeExit(const ScopeExit &) = delete;
  ScopeExit & operator=(const ScopeExit &) = delete;

private:
  F f_;
};

template<typename F>
ScopeExit<F> MakeScopeExit(F && f)
{
  return ScopeExit<F>(std::forward<F>(f));
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

  // --- Build prompt tensors on GPU ---

  // Track GPU allocations so we can release them on any early-failure path.
  // Ownership is transferred to the NitrosTensor release callbacks once
  // allocations_released is set to true below.
  std::vector<void *> pending_allocations;
  bool allocations_released = false;
  auto alloc_guard = MakeScopeExit(
    [&] {
      if (!allocations_released) {
        for (void * p : pending_allocations) {
          if (p) {cudaFree(p);}
        }
      }
    });

  auto try_alloc = [&](void ** ptr, size_t bytes) -> bool {
      const cudaError_t err = cudaMalloc(ptr, bytes);
      if (err != cudaSuccess) {
        RCLCPP_ERROR(
          get_logger(),
          "cudaMalloc failed for %zu bytes: %s",
          bytes, cudaGetErrorString(err));
        *ptr = nullptr;
        return false;
      }
      pending_allocations.push_back(*ptr);
      return true;
    };

  // Points tensor: shape [batch_size, num_points, 2]
  uint32_t points_buffer_size = batch_size * num_points * 2 * sizeof(float);
  void * points_gpu = nullptr;
  if (!try_alloc(&points_gpu, points_buffer_size)) {
    return;
  }
  cudaMemcpyAsync(
    points_gpu, prompt_vec.data(), points_buffer_size,
    cudaMemcpyHostToDevice, cuda_stream_);

  // Labels tensor: shape [batch_size, num_points]
  uint32_t labels_buffer_size = batch_size * num_points * sizeof(float);
  void * labels_gpu = nullptr;
  if (!try_alloc(&labels_gpu, labels_buffer_size)) {
    return;
  }
  cudaMemcpyAsync(
    labels_gpu, label_vec.data(), labels_buffer_size,
    cudaMemcpyHostToDevice, cuda_stream_);

  // has_input_mask tensor: shape [1]
  std::vector<float> has_mask_data = {has_input_mask_ ? 1.0f : 0.0f};
  void * has_mask_gpu = nullptr;
  if (!try_alloc(&has_mask_gpu, sizeof(float))) {
    return;
  }
  cudaMemcpyAsync(
    has_mask_gpu, has_mask_data.data(), sizeof(float),
    cudaMemcpyHostToDevice, cuda_stream_);

  // orig_img_dims tensor: shape [2]
  std::vector<float> img_size_data = {
    static_cast<float>(orig_img_dims_[0]),
    static_cast<float>(orig_img_dims_[1])};
  void * img_size_gpu = nullptr;
  if (!try_alloc(&img_size_gpu, 2 * sizeof(float))) {
    return;
  }
  cudaMemcpyAsync(
    img_size_gpu, img_size_data.data(), 2 * sizeof(float),
    cudaMemcpyHostToDevice, cuda_stream_);

  // --- Forward tensors from image and mask inputs ---
  // Collect forwarded tensors: copy GPU data from inputs
  struct ForwardedTensor
  {
    std::string name;
    nitros::NitrosTensorShape shape;
    nitros::NitrosDataType dtype;
    void * gpu_data;
  };
  std::vector<ForwardedTensor> forwarded_tensors;

  auto forward_tensors_from_msg =
    [&](const nitros::NitrosTensorList & msg) -> bool {
      for (const auto & tensor : msg.get_tensors()) {
        size_t tensor_size = tensor.element_count() * tensor.bytes_per_element();
        void * gpu_copy = nullptr;
        if (!try_alloc(&gpu_copy, tensor_size)) {
          return false;
        }
        cudaMemcpyAsync(
          gpu_copy, tensor.get_read_handle(msg.get_stream()).get_ptr(), tensor_size,
          cudaMemcpyDeviceToDevice, cuda_stream_);
        forwarded_tensors.push_back(
          ForwardedTensor{tensor.get_name(), tensor.shape(),
            tensor.data_type(), gpu_copy});
      }
      return true;
    };

  if (!forward_tensors_from_msg(*image_tensor)) {
    return;
  }
  if (!forward_tensors_from_msg(*mask_tensor)) {
    return;
  }

  // Synchronize before building output
  cudaStreamSynchronize(cuda_stream_);

  // --- Build composite output tensor list ---
  std_msgs::msg::Header header;
  header.stamp = prompts->header.stamp;
  header.frame_id = prompts->header.frame_id;

  auto builder = nitros::NitrosTensorListBuilder().WithHeader(header);

  // Add prompt-derived tensors
  builder.AddTensor(
    "points",
    nitros::NitrosTensorBuilder()
    .WithShape(nitros::NitrosTensorShape(
      {static_cast<int32_t>(batch_size),
        static_cast<int32_t>(num_points), 2}))
    .WithDataType(nitros::NitrosDataType::kFloat32)
    .WithData(points_gpu)
    .WithReleaseCallback([points_gpu]() {cudaFree(points_gpu);})
    .Build());

  builder.AddTensor(
    "labels",
    nitros::NitrosTensorBuilder()
    .WithShape(nitros::NitrosTensorShape(
      {static_cast<int32_t>(batch_size),
        static_cast<int32_t>(num_points)}))
    .WithDataType(nitros::NitrosDataType::kFloat32)
    .WithData(labels_gpu)
    .WithReleaseCallback([labels_gpu]() {cudaFree(labels_gpu);})
    .Build());

  builder.AddTensor(
    "orig_img_dims",
    nitros::NitrosTensorBuilder()
    .WithShape(nitros::NitrosTensorShape({2}))
    .WithDataType(nitros::NitrosDataType::kFloat32)
    .WithData(img_size_gpu)
    .WithReleaseCallback([img_size_gpu]() {cudaFree(img_size_gpu);})
    .Build());

  builder.AddTensor(
    "has_input_mask",
    nitros::NitrosTensorBuilder()
    .WithShape(nitros::NitrosTensorShape({1}))
    .WithDataType(nitros::NitrosDataType::kFloat32)
    .WithData(has_mask_gpu)
    .WithReleaseCallback([has_mask_gpu]() {cudaFree(has_mask_gpu);})
    .Build());

  // Add forwarded tensors from image and mask inputs
  for (auto & ft : forwarded_tensors) {
    void * data = ft.gpu_data;
    builder.AddTensor(
      ft.name,
      nitros::NitrosTensorBuilder()
      .WithShape(ft.shape)
      .WithDataType(ft.dtype)
      .WithData(data)
      .WithReleaseCallback([data]() {cudaFree(data);})
      .Build());
  }

  // All GPU buffers are now owned by the tensor release callbacks; prevent
  // the guard from double-freeing them.
  allocations_released = true;
  output_pub_->publish(std::move(builder.Build()));
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
