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

#ifndef ISAAC_ROS_SEGMENT_ANYTHING2__SEGMENT_ANYTHING2_DATA_ENCODER_NODE_HPP_
#define ISAAC_ROS_SEGMENT_ANYTHING2__SEGMENT_ANYTHING2_DATA_ENCODER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>
#include "isaac_ros_common/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "isaac_ros_nitros_detection2_d_array_type/nitros_detection2_d_array.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_segment_anything2/segment_anything2_state_manager.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_segment_anything2_interfaces/srv/add_objects.hpp"
#include "isaac_ros_segment_anything2_interfaces/srv/remove_object.hpp"
#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything2
{

class SegmentAnything2DataEncoderNode : public rclcpp::Node
{
public:
  explicit SegmentAnything2DataEncoderNode(
    const rclcpp::NodeOptions options = rclcpp::NodeOptions()
  );
  ~SegmentAnything2DataEncoderNode();

private:
  // QoS settings
  rclcpp::QoS image_qos_;
  rclcpp::QoS memory_qos_;
  rclcpp::QoS encoded_data_qos_;

  // Callbacks for subscribers
  void ImageCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);
  void MemoryCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);

  // Service callback for adding objects
  void AddObjectsCallback(
    const std::shared_ptr<
      isaac_ros_segment_anything2_interfaces::srv::AddObjects::Request> request,
    std::shared_ptr<
      isaac_ros_segment_anything2_interfaces::srv::AddObjects::Response> response);

  // Service callback to remove object
  void RemoveObjectCallback(
    const std::shared_ptr<
      isaac_ros_segment_anything2_interfaces::srv::RemoveObject::Request> request,
    std::shared_ptr<
      isaac_ros_segment_anything2_interfaces::srv::RemoveObject::Response> response);

  // Publisher for output NitrosTensorList messages
  std::shared_ptr<
    nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosTensorList>> encoded_data_pub_;
  // Subscribers
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>> image_sub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>> memory_sub_;

  // Service for adding objects to the segmentation model
  rclcpp::Service<
    isaac_ros_segment_anything2_interfaces::srv::AddObjects>::SharedPtr add_objects_srv_;

  // Service for adding objects to the segmentation model
  rclcpp::Service<
    isaac_ros_segment_anything2_interfaces::srv::RemoveObject>::SharedPtr remove_object_srv_;
  // Maximum number of objects that can be added to the state manager.
  // Required because post processor always caps the number of objects.
  int32_t max_num_objects_;

  std::vector<int64_t> orig_img_dims_param_;
  int32_t * original_size_buffer_;
  std::unique_ptr<SAM2StateManager> sam2_state_manager_;
  cudaStream_t stream_;
};

}  // namespace segment_anything2
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_SEGMENT_ANYTHING2__SEGMENT_ANYTHING2_DATA_ENCODER_NODE_HPP_
