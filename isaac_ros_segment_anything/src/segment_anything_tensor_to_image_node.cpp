// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_segment_anything/segment_anything_binarize_tensor.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

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

// Size in bytes of a single element from the tensor's DLPack dtype fields.
static inline uint64_t BytesPerElement(const tensor_msgs::msg::ExperimentalTensor & tensor)
{
  const uint64_t bits =
    static_cast<uint64_t>(tensor.dtype_bits) * static_cast<uint64_t>(tensor.dtype_lanes);
  if (bits == 0 || bits % 8 != 0) {
    throw std::runtime_error(
            "Unsupported tensor dtype: dtype_bits=" + std::to_string(tensor.dtype_bits) +
            " dtype_lanes=" + std::to_string(tensor.dtype_lanes));
  }
  return bits / 8;
}

TensorToImageNode::TensorToImageNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("tensor_to_image", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, kDefaultQoS, "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, kDefaultQoS, "output_qos")}
{
  // Initialize subscriber
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  tensor_list_sub_ = create_subscription<isaac_ros_tensor_msgs::msg::TensorList>(
    "segmentation_tensor", input_qos_,
    std::bind(&TensorToImageNode::TensorListCallback, this, std::placeholders::_1),
    sub_options);

  // Initialize publisher
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  binary_mask_pub_ = create_publisher<sensor_msgs::msg::Image>(
    "binary_mask", output_qos_, pub_options);

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
  const isaac_ros_tensor_msgs::msg::TensorList::ConstSharedPtr & tensor_list_msg)
{
  try {
    // Get all tensors and verify we have at least one
    if (tensor_list_msg->tensors.empty()) {
      throw std::runtime_error("TensorList is empty");
    }

    // Get the first tensor in the list
    const auto & tensor = tensor_list_msg->tensors.at(0);

    if (tensor.shape.size() != 4) {
      std::string rank_str = std::to_string(tensor.shape.size());
      throw std::runtime_error("Tensor has incorrect rank, expected rank 4 but got " + rank_str);
    }

    // Get height and width, the input is a tensor of shape [batch_size, 1, height, width]
    int height = static_cast<int>(tensor.shape[2]);
    int width = static_cast<int>(tensor.shape[3]);
    int batch_size = static_cast<int>(tensor.shape[0]);
    int num_channels = static_cast<int>(tensor.shape[1]);

    // Get size for tensor data elements in bytes
    uint64_t element_size = BytesPerElement(tensor);

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
    // which contains a tensor of shape [1, height, width] of type uint8.
    if (element_size != 1) {
      throw std::runtime_error(
              "Tensor has incorrect element size, expected 1 byte per element but got " + \
              std::to_string(element_size));
    }

    // Build the header from the upstream tensor list before allocating the output
    // image so it can be assigned to the image below.
    std_msgs::msg::Header header;
    header.frame_id = tensor_list_msg->header.frame_id;
    header.stamp.sec = tensor_list_msg->header.stamp.sec;
    header.stamp.nanosec = tensor_list_msg->header.stamp.nanosec;

    // Allocate the output image with a cuda_buffer-backed payload. The WriteHandle
    // obtained from the buffer records a CUDA event on destruction so downstream
    // consumers automatically wait on this write, replacing the need for a
    // producer-side cudaStreamSynchronize before publish.
    const size_t image_bytes =
      static_cast<size_t>(height) * static_cast<size_t>(width) * element_size;

    auto mask_image = std::make_unique<sensor_msgs::msg::Image>();
    mask_image->header = header;
    mask_image->height = height;
    mask_image->width = width;
    mask_image->encoding = GetImageEncoding(element_size);
    mask_image->is_bigendian = 0;
    mask_image->step = static_cast<uint32_t>(width * element_size);
    mask_image->data = cuda_buffer_backend::allocate_buffer(image_bytes);

    BoundingBox bbox;
    {
      auto wh = cuda_buffer_backend::from_output_buffer(mask_image->data, stream_);
      uint8_t * mask_ptr = wh.get_ptr();

      // Copy tensor data into the freshly allocated buffer rather than aliasing the
      // input tensor's memory, which is owned by the upstream producer.
      auto tensor_read_handle = cuda_buffer_backend::from_input_buffer(tensor.data, stream_);
      CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
          mask_ptr, tensor_read_handle.get_ptr() + tensor.byte_offset, image_bytes,
          cudaMemcpyDeviceToDevice, stream_),
        "Failed to copy tensor data to GPU memory");

      BinarizeTensorOnGPU(mask_ptr, image_bytes, stream_);
      FindBoundingBoxOnGPU(mask_ptr, width, height, &bbox, stream_);
    }  // WriteHandle destructor records the producer-side CUDA event here.

    // Sync still required: FindBoundingBoxOnGPU issues async D2H copies into the
    // host-side `bbox` struct that is read immediately below.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "Failed to synchronize CUDA stream");

    binary_mask_pub_->publish(std::move(mask_image));
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
