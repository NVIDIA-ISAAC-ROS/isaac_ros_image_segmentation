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

#include "isaac_ros_segment_anything/segment_anything_tensor_to_image_node.hpp"

#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_segment_anything/segment_anything_binarize_tensor.hpp"

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

namespace
{
constexpr const char kDefaultQoS[] = "DEFAULT";
}  // namespace

std::string GetImageEncoding(const uint64_t element_size)
{
  if (element_size == sizeof(uint8_t)) {
    return sensor_msgs::image_encodings::MONO8;
  } else {
    throw std::runtime_error(
            "Unsupported encoding type for element size calculation: " + std::to_string(
              element_size));
  }
}

TensorToImageNode::TensorToImageNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("tensor_to_image", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, kDefaultQoS, "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, kDefaultQoS, "output_qos")}
{
  // Initialize NITROS subscriber
  tensor_list_sub_ =
    std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
    this,
    "segmentation_tensor",
    nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_t::supported_type_name,
    std::bind(
      &TensorToImageNode::TensorListCallback, this,
      std::placeholders::_1),
    nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig(),
    input_qos_);

  // Initialize NITROS publisher
  binary_mask_pub_ =
    std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosImage>>(
    this,
    "binary_mask",
    nvidia::isaac_ros::nitros::nitros_image_mono8_t::supported_type_name,
    nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig(),
    output_qos_);

  // Initialize standard ROS publisher for detections
  detection_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    "detection_array", output_qos_);

  // Initialize CUDA stream
  CHECK_CUDA_ERROR(
    ::nvidia::isaac_ros::common::initNamedCudaStream(
      stream_, "isaac_ros_tensor_to_image_node"),
    "Error initializing CUDA stream");

  RCLCPP_INFO(get_logger(), "[TensorToImageNode] Initialized");
}

void TensorToImageNode::TensorListCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & tensor_list_view)
{
  try {
    // Get all tensors and verify we have at least one
    const auto tensor_views = tensor_list_view.GetAllTensor();
    if (tensor_views.empty()) {
      throw std::runtime_error("TensorList is empty");
    }

    // Get the first tensor in the list
    const auto & tensor_view = tensor_views[0];

    if (tensor_view.GetRank() != 4) {
      std::string rank_str = std::to_string(tensor_view.GetRank());
      throw std::runtime_error("Tensor has incorrect rank, expected rank 4 but got " + rank_str);
    }

    // Get height and width, the input is a tensor of shape [batch_size, 1, height, width]
    int height = static_cast<int>(tensor_view.GetDimension(2));
    int width = static_cast<int>(tensor_view.GetDimension(3));
    int batch_size = static_cast<int>(tensor_view.GetDimension(0));
    int num_channels = static_cast<int>(tensor_view.GetDimension(1));

    // Get size for tensor data elements in bytes
    uint64_t element_size = tensor_view.GetBytesPerElement();

    RCLCPP_DEBUG(
      get_logger(), "Width: %d, height: %d, element_size: %lu",
      width, height, element_size);

    // Check batch size is 1
    if (batch_size != 1) {
      std::string batch_size_str = std::to_string(batch_size);
      throw std::runtime_error(
              "Only batch size 1 is supported (got batch size " + batch_size_str + ")");
    }

    if (num_channels != 1) {
      throw std::runtime_error(
              "Only 1 channel is supported (got " + std::to_string(num_channels) + ")");
    }

    // Also error out if bytes per element is not 1, because we are expecting a tensorlist
    // which contains a tensor of shape [1, height, width] of type uint8, since managed
    // nitros publishers can only publish uint8 tensors, we have to add this constraint.
    if (element_size != 1) {
      throw std::runtime_error(
              "Tensor has incorrect element size, expected 1 byte per element but got " + \
              std::to_string(element_size));
    }

    // Create a new GPU memory for this new image, with the stream object
    void * gpu_buffer = nullptr;
    CHECK_CUDA_ERROR(
      cudaMallocAsync(&gpu_buffer, height * width * element_size, stream_),
      "Failed to allocate GPU memory");

    // Copy the tensor data to the GPU memory, this is required since NitrosPUblishers and
    // Subscribers rely on the underlying GXF memory to manage the memory. But this doesn't work
    // when we take the raw pointer from the tensor view and pass it to the NitrosImageBuilder.
    // This is because we break the assumotion that the memory will always be part of a single
    // type of GXF object (e.g. VideoBuffer, Tensor, etc).
    CHECK_CUDA_ERROR(
      cudaMemcpyAsync(
        gpu_buffer, tensor_view.GetBuffer(), height * width * element_size,
        cudaMemcpyDeviceToDevice, stream_),
      "Failed to copy tensor data to GPU memory");

    // Binarize the tensor data on GPU
    BinarizeTensorOnGPU(
      static_cast<uint8_t *>(gpu_buffer),
      height * width * element_size, stream_);

    // Find bounding box around non-zero values
    BoundingBox bbox;
    FindBoundingBoxOnGPU(
      static_cast<uint8_t *>(gpu_buffer), width, height, &bbox, stream_);

    // Sync the stream, before publishing it to prevent memory corruption errors downstream.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "Failed to synchronize CUDA stream");

    // Create a header for the mask
    std_msgs::msg::Header header;
    header.frame_id = tensor_list_view.GetFrameId();
    header.stamp.sec = tensor_list_view.GetTimestampSeconds();
    header.stamp.nanosec = tensor_list_view.GetTimestampNanoseconds();

    // Create and publish the image directly from tensor data
    // Note: const_cast is safe here as the data won't be modified, just read
    auto mask_image = nvidia::isaac_ros::nitros::NitrosImageBuilder()
      .WithHeader(header)
      .WithDimensions(height, width)
      .WithEncoding(GetImageEncoding(tensor_view.GetBytesPerElement()))
      .WithGpuData(gpu_buffer)
      .Build();
    binary_mask_pub_->publish(mask_image);
    RCLCPP_DEBUG(
      get_logger(),
      "Published image with dimensions %dx%d", width, height);

    // Create and publish the detection2d array
    vision_msgs::msg::Detection2DArray detection_array_msg;
    detection_array_msg.header = header;

    // Create a detection message
    vision_msgs::msg::Detection2D detection_msg;
    detection_msg.header = header;

    // Calculate center and size for the bounding box
    float center_x = (bbox.min_x + bbox.max_x) / 2.0f;
    float center_y = (bbox.min_y + bbox.max_y) / 2.0f;
    float size_x = abs(bbox.max_x - bbox.min_x) + 1.0f;
    float size_y = abs(bbox.max_y - bbox.min_y) + 1.0f;

    // Set the bounding box
    detection_msg.bbox.center.position.x = center_x;
    detection_msg.bbox.center.position.y = center_y;
    detection_msg.bbox.center.theta = 0.0;
    detection_msg.bbox.size_x = size_x;
    detection_msg.bbox.size_y = size_y;

    // Add a default hypothesis
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = "mask";
    hypothesis.hypothesis.score = 1.0;
    detection_msg.results.push_back(hypothesis);

    // Add the detection to the array and publish
    detection_array_msg.detections.push_back(detection_msg);
    detection_pub_->publish(detection_array_msg);

    RCLCPP_DEBUG(
      get_logger(),
      "Published detection with bbox: (%d,%d) to (%d,%d)",
      bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      get_logger(),
      "Error in TensorListCallback: %s", e.what());
  }
}

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::segment_anything::TensorToImageNode)
