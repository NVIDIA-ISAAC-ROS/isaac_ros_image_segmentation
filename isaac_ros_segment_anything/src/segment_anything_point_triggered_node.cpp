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

#include "isaac_ros_nitros_topic_tools/isaac_ros_nitros_topic_tools_common.hpp"
#include "isaac_ros_segment_anything/segment_anything_point_triggered_node.hpp"

#include "rclcpp/rclcpp.hpp"

#include <isaac_ros_common/qos.hpp>

namespace
{
constexpr double kDetectionSize = 10.0;
}

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{

namespace
{
constexpr const char kDefaultQoS[] = "DEFAULT";
}

SegmentAnythingPointTriggeredNode::SegmentAnythingPointTriggeredNode(
  const rclcpp::NodeOptions & options)
: rclcpp::Node("point_triggered", options)
{
  mode_ = declare_parameter<std::string>("mode", modeToStringMap.at(CameraDropMode::Mono));
  is_sam2_ = declare_parameter<bool>("is_sam2", false);
  is_triggered_ = false;
  depth_format_string_ =
    declare_parameter<std::string>("depth_format_string", "nitros_image_32FC1");
  sync_queue_size_ = declare_parameter<int>("sync_queue_size", 10);
  input_queue_size_ = declare_parameter<int>("input_queue_size", 10);
  output_queue_size_ = declare_parameter<int>("output_queue_size", 10);
  max_rate_hz_ = declare_parameter<double>("max_rate_hz", 2.0);  // Default to 2 Hz
  service_call_timeout_ = declare_parameter<int>("service_call_timeout", 5);
  service_discovery_timeout_ = declare_parameter<int>("service_discovery_timeout", 5);

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, kDefaultQoS, "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, kDefaultQoS, "output_qos").keep_last(output_queue_size_);
  const rmw_qos_profile_t input_qos_profile = input_qos.get_rmw_qos_profile();

  // Initialize last point time
  last_point_time_ = std::chrono::steady_clock::now();

  // Create mutually exclusive callback group for the service clients so service calls cannot be
  // executed in parallel to itself or other service calls.
  client_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Create reentrant callback group for topic subscriptions for parallel execution of callbacks.
  subscription_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = subscription_cb_group_;

  // Point subscriber
  point_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
    "point", input_qos,
    std::bind(&SegmentAnythingPointTriggeredNode::point_callback, this, std::placeholders::_1),
    sub_options);

  // AddObjects service client
  add_objects_client_ = this->create_client<AddObjectsSrv>(
    "add_objects", rclcpp::ServicesQoS(), client_callback_group_);

  // RemoveObject service client
  remove_object_client_ = this->create_client<RemoveObjectSrv>(
    "remove_object", rclcpp::ServicesQoS(), client_callback_group_);

  // Initialize common subscribers and publishers for all modes
  image_sub_1_.subscribe(this, "image_1", input_qos_profile, sub_options);
  camera_info_sub_1_.subscribe(this, "camera_info_1", input_qos_profile, sub_options);
  image_pub_1_ = std::make_shared<
    nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>(
    this, "image_1_triggered",
    nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name,
    nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig(), output_qos);
  camera_info_pub_1_ = this->create_publisher<
    sensor_msgs::msg::CameraInfo>("camera_info_1_triggered", output_qos);
  detection_pub_ = this->create_publisher<
    vision_msgs::msg::Detection2DArray>("detection_2d", output_qos);

  if (mode_ == modeToStringMap.at(CameraDropMode::Mono)) {
    // Mode 0: Camera + CameraInfo (mono)
    // Initialize sync policy
    exact_sync_mode_0_ = std::make_shared<ExactSyncMode0>(
      ExactPolicyMode0(sync_queue_size_), image_sub_1_,
      camera_info_sub_1_);
    using namespace std::placeholders;
    exact_sync_mode_0_->registerCallback(
      std::bind(&SegmentAnythingPointTriggeredNode::sync_callback_mode_0, this, _1, _2));
  } else if (mode_ == modeToStringMap.at(CameraDropMode::Stereo)) {
    // Mode 1: Camera + CameraInfo + Camera + CameraInfo (stereo)
    // Initialize second mono subscribers and publishers
    image_sub_2_.subscribe(this, "image_2", input_qos_profile, sub_options);
    camera_info_sub_2_.subscribe(this, "camera_info_2", input_qos_profile, sub_options);
    image_pub_2_ = std::make_shared<
      nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "image_2_triggered",
      nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name,
      nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig(), output_qos);
    camera_info_pub_2_ = this->create_publisher<
      sensor_msgs::msg::CameraInfo>("camera_info_2_triggered", output_qos);
    // Initialize sync policy
    exact_sync_mode_1_ = std::make_shared<ExactSyncMode1>(
      ExactPolicyMode1(sync_queue_size_), image_sub_1_, camera_info_sub_1_, image_sub_2_,
      camera_info_sub_2_);
    using namespace std::placeholders;
    exact_sync_mode_1_->registerCallback(
      std::bind(&SegmentAnythingPointTriggeredNode::sync_callback_mode_1, this, _1, _2, _3, _4));
  } else if (mode_ == modeToStringMap.at(CameraDropMode::MonoDepth)) {
    // Mode 2: Camera + CameraInfo + Depth (mono+depth)
    // Initialize depth subscriber and publisher
    depth_sub_.subscribe(
      this, "depth_1", input_qos_profile, sub_options,
      depth_format_string_);
    depth_pub_ = std::make_shared<
      nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "depth_1_triggered", depth_format_string_);
    // Initialize sync policy
    exact_sync_mode_2_ = std::make_shared<ExactSyncMode2>(
      ExactPolicyMode2(sync_queue_size_), image_sub_1_, camera_info_sub_1_, depth_sub_);
    using namespace std::placeholders;
    exact_sync_mode_2_->registerCallback(
      std::bind(&SegmentAnythingPointTriggeredNode::sync_callback_mode_2, this, _1, _2, _3));
  } else {
    RCLCPP_ERROR(get_logger(), "Invalid mode: %s", mode_.c_str());
  }
}

void SegmentAnythingPointTriggeredNode::point_callback(
  const geometry_msgs::msg::Point::SharedPtr point_msg)
{
  RCLCPP_DEBUG(
    get_logger(), "Received point: (%f, %f, %f)",
    point_msg->x, point_msg->y, point_msg->z);

  // Check rate limit
  if (!check_rate_limit()) {
    RCLCPP_DEBUG(get_logger(), "Point ignored due to rate limiting");
    return;
  }

  // Process point and publish data
  process_point(point_msg);
}

vision_msgs::msg::Detection2DArray SegmentAnythingPointTriggeredNode::point_to_detection2d_array(
  const geometry_msgs::msg::Point & point,
  const rclcpp::Time & timestamp)
{
  vision_msgs::msg::Detection2DArray detections_array;
  detections_array.header.stamp = timestamp;

  vision_msgs::msg::Detection2D detection;
  detection.header.stamp = timestamp;
  detection.bbox.center.position.x = point.x;
  detection.bbox.center.position.y = point.y;

  // Set size to a small value since it's a point click
  detection.bbox.size_x = kDetectionSize;
  detection.bbox.size_y = kDetectionSize;

  detections_array.detections.push_back(detection);

  return detections_array;
}

bool SegmentAnythingPointTriggeredNode::call_add_objects(
  const geometry_msgs::msg::Point initial_hint)
{
  // Wait for add objects service to be available
  if (!add_objects_client_->wait_for_service(std::chrono::seconds(service_discovery_timeout_))) {
    RCLCPP_ERROR(get_logger(), "AddObjects service not available");
    return false;
  }

  // Create a request for the AddObjects service
  auto request = std::make_shared<AddObjectsSrv::Request>();
  request->request_header = std_msgs::msg::Header();
  request->request_header.stamp = this->now();

  vision_msgs::msg::Point2D initial_hint_point;
  initial_hint_point.x = initial_hint.x;
  initial_hint_point.y = initial_hint.y;

  request->point_coords.push_back(initial_hint_point);
  request->point_object_ids.push_back(std::to_string(0));
  request->point_labels.push_back(1);  // 1 for foreground

  // Call the AddObjects service with the request
  auto future = add_objects_client_->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(service_call_timeout_)) == std::future_status::timeout) {
    RCLCPP_ERROR(get_logger(), "AddObjects service call timed out");
    return false;
  }

  // Get the response from the AddObjects service
  auto response = future.get();
  if (!response->success) {
    RCLCPP_ERROR(get_logger(), "AddObjects service failed: %s", response->message.c_str());
    return false;
  }

  return true;
}

bool SegmentAnythingPointTriggeredNode::call_remove_object()
{
  // Wait for remove object service to be available
  if (!remove_object_client_->wait_for_service(std::chrono::seconds(service_discovery_timeout_))) {
    RCLCPP_ERROR(get_logger(), "RemoveObject service not available");
    return false;
  }

  // Create a request for the RemoveObject service
  auto request = std::make_shared<RemoveObjectSrv::Request>();
  request->request_header = std_msgs::msg::Header();
  request->request_header.stamp = this->now();
  request->object_id = std::to_string(0);

  // Call the RemoveObject service
  auto future = remove_object_client_->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(service_call_timeout_)) == std::future_status::timeout) {
    RCLCPP_ERROR(get_logger(), "RemoveObject service call timed out");
    return false;
  }

  // Get the response from the RemoveObject service
  auto response = future.get();
  if (!response->success) {
    RCLCPP_ERROR(get_logger(), "RemoveObject service failed: %s", response->message.c_str());
    return false;
  }

  return true;
}

bool SegmentAnythingPointTriggeredNode::check_rate_limit()
{
  auto current_time = std::chrono::steady_clock::now();
  auto time_since_last = current_time - last_point_time_;
  double seconds_since_last = std::chrono::duration<double>(time_since_last).count();

  if (seconds_since_last < (1.0 / max_rate_hz_)) {
    return false;  // Rate limit exceeded
  }

  // Update last_point_time_
  last_point_time_ = current_time;
  return true;
}

void SegmentAnythingPointTriggeredNode::sync_callback_mode_0(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_ptr,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_ptr)
{
  // Store the latest synchronized data
  latest_mode_0_data_.image = image_ptr;
  latest_mode_0_data_.camera_info = camera_info_ptr;
  latest_mode_0_data_.timestamp = std::chrono::steady_clock::now();
  latest_mode_0_data_.valid = true;
}

void SegmentAnythingPointTriggeredNode::sync_callback_mode_1(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_1_ptr,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_1_ptr,
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_2_ptr,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_2_ptr)
{
  // Store the latest synchronized stereo data
  latest_mode_1_data_.image_1 = image_1_ptr;
  latest_mode_1_data_.camera_info_1 = camera_info_1_ptr;
  latest_mode_1_data_.image_2 = image_2_ptr;
  latest_mode_1_data_.camera_info_2 = camera_info_2_ptr;
  latest_mode_1_data_.timestamp = std::chrono::steady_clock::now();
  latest_mode_1_data_.valid = true;
}

void SegmentAnythingPointTriggeredNode::sync_callback_mode_2(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_ptr,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_ptr,
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & depth_ptr)
{
  // Store the latest synchronized mono+depth data
  latest_mode_2_data_.image = image_ptr;
  latest_mode_2_data_.camera_info = camera_info_ptr;
  latest_mode_2_data_.depth = depth_ptr;
  latest_mode_2_data_.timestamp = std::chrono::steady_clock::now();
  latest_mode_2_data_.valid = true;
}

void SegmentAnythingPointTriggeredNode::process_point(
  const geometry_msgs::msg::Point::SharedPtr point_msg)
{
  // If using SAM2, call required services to segment the object
  if (is_sam2_) {
    // If previously triggered, remove the object first
    if (is_triggered_ && !call_remove_object()) {
      return;
    }

    // Call the AddObjects service
    if (!call_add_objects(*point_msg)) {
      return;
    }
  }

  // Process point based on current mode
  if (mode_ == modeToStringMap.at(CameraDropMode::Mono)) {
    if (!latest_mode_0_data_.valid) {
      RCLCPP_WARN(get_logger(), "No valid mono camera data available yet");
      return;
    }

    // Publish image and camera info
    image_pub_1_->publish(*latest_mode_0_data_.image);
    camera_info_pub_1_->publish(*latest_mode_0_data_.camera_info);

    // If using SAM1, convert point to Detection2DArray with camera info timestamp and publish
    if (!is_sam2_) {
      auto now = rclcpp::Time(latest_mode_0_data_.camera_info->header.stamp);
      auto detection_msg = point_to_detection2d_array(*point_msg, now);
      detection_pub_->publish(detection_msg);
    }

    RCLCPP_DEBUG(get_logger(), "Published mono camera data");
  } else if (mode_ == modeToStringMap.at(CameraDropMode::Stereo)) {
    if (!latest_mode_1_data_.valid) {
      RCLCPP_WARN(get_logger(), "No valid stereo camera data available yet");
      return;
    }

    // Publish images and camera info
    image_pub_1_->publish(*latest_mode_1_data_.image_1);
    camera_info_pub_1_->publish(*latest_mode_1_data_.camera_info_1);
    image_pub_2_->publish(*latest_mode_1_data_.image_2);
    camera_info_pub_2_->publish(*latest_mode_1_data_.camera_info_2);

    // If using SAM1, convert point to Detection2DArray with camera info timestamp and publish
    if (!is_sam2_) {
      auto now = rclcpp::Time(latest_mode_1_data_.camera_info_1->header.stamp);
      auto detection_msg = point_to_detection2d_array(*point_msg, now);
      detection_pub_->publish(detection_msg);
    }

    RCLCPP_DEBUG(get_logger(), "Published stereo camera data");
  } else if (mode_ == modeToStringMap.at(CameraDropMode::MonoDepth)) {
    if (!latest_mode_2_data_.valid) {
      RCLCPP_WARN(get_logger(), "No valid mono+depth camera data available yet");
      return;
    }

    // Publish image, camera info, and depth
    image_pub_1_->publish(*latest_mode_2_data_.image);
    camera_info_pub_1_->publish(*latest_mode_2_data_.camera_info);
    depth_pub_->publish(*latest_mode_2_data_.depth);

    // If using SAM1, convert point to Detection2DArray with camera info timestamp and publish
    if (!is_sam2_) {
      auto now = rclcpp::Time(latest_mode_2_data_.camera_info->header.stamp);
      auto detection_msg = point_to_detection2d_array(*point_msg, now);
      detection_pub_->publish(detection_msg);
    }

    RCLCPP_DEBUG(get_logger(), "Published mono+depth camera data");
  }

  // Set the is triggered flag to true
  is_triggered_ = true;
}

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::segment_anything::SegmentAnythingPointTriggeredNode)
