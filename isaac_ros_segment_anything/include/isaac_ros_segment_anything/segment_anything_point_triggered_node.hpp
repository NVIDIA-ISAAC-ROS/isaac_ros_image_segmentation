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

#pragma once

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>

#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_detection2_d_array_type/nitros_detection2_d_array.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"
#include "isaac_ros_segment_anything2_interfaces/srv/add_objects.hpp"
#include "isaac_ros_segment_anything2_interfaces/srv/remove_object.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{
/**
 * @brief SegmentAnythingPointTriggeredNode class implements a node that
 *        listens for point messages and synchronizes them with camera streams.
 *        When a point is received, it triggers the publishing of synchronized
 *        image, camera info, and depth data (if available), and also converts
 *        the point to a Detection2DArray message.
 *        The node also limits the publishing rate to a maximum frequency.
 *        It supports the same three modes as NitrosCameraDropNode:
 *        - Mode 0(mono): Camera + CameraInfo
 *        - Mode 1(stereo): Camera + CameraInfo + Camera + CameraInfo
 *        - Mode 2(mono+depth): Camera + CameraInfo + Depth
 */
class SegmentAnythingPointTriggeredNode : public rclcpp::Node
{
public:
  /**
   * @brief Constructor for SegmentAnythingPointTriggeredNode class.
   * @param options The node options.
   */
  explicit SegmentAnythingPointTriggeredNode(const rclcpp::NodeOptions & options);

private:
  using AddObjectsSrv = isaac_ros_segment_anything2_interfaces::srv::AddObjects;
  using RemoveObjectSrv = isaac_ros_segment_anything2_interfaces::srv::RemoveObject;

  // Subscribers
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr point_sub_;
  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosImageView>
  image_sub_1_;
  ::message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_1_;
  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosImageView>
  image_sub_2_;
  ::message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_2_;
  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosImageView>
  depth_sub_;

  // Publishers
  std::shared_ptr<
    nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>
  image_pub_1_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_1_;
  std::shared_ptr<
    nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>
  image_pub_2_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_2_;
  std::shared_ptr<
    nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>
  depth_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;

  // Service clients
  rclcpp::Client<AddObjectsSrv>::SharedPtr add_objects_client_;
  rclcpp::Client<RemoveObjectSrv>::SharedPtr remove_object_client_;

  // Callback groups
  rclcpp::CallbackGroup::SharedPtr client_callback_group_;
  rclcpp::CallbackGroup::SharedPtr subscription_cb_group_;

  // Exact message sync policy
  using ExactPolicyMode0 = ::message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage, sensor_msgs::msg::CameraInfo>;
  using ExactSyncMode0 = ::message_filters::Synchronizer<ExactPolicyMode0>;
  std::shared_ptr<ExactSyncMode0> exact_sync_mode_0_;  // Exact sync mode 0

  using ExactPolicyMode1 = ::message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage, sensor_msgs::msg::CameraInfo,
    nvidia::isaac_ros::nitros::NitrosImage, sensor_msgs::msg::CameraInfo>;
  using ExactSyncMode1 = ::message_filters::Synchronizer<ExactPolicyMode1>;
  std::shared_ptr<ExactSyncMode1> exact_sync_mode_1_;  // Exact sync mode 1

  using ExactPolicyMode2 = ::message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage, sensor_msgs::msg::CameraInfo,
    nvidia::isaac_ros::nitros::NitrosImage>;
  using ExactSyncMode2 = ::message_filters::Synchronizer<ExactPolicyMode2>;
  std::shared_ptr<ExactSyncMode2> exact_sync_mode_2_;  // Exact sync mode 2

  /**
   * @brief Callback function for point subscription.
   * @param point_msg The received point message.
   */
  void point_callback(const geometry_msgs::msg::Point::SharedPtr point_msg);

  /**
   * @brief Convert a point to a Detection2DArray message.
   * @param point The point message to convert.
   * @param timestamp The timestamp to use for the detection message.
   * @return The created Detection2DArray message.
   */
  vision_msgs::msg::Detection2DArray point_to_detection2d_array(
    const geometry_msgs::msg::Point & point,
    const rclcpp::Time & timestamp);

  /**
   * @brief Check if the rate limit allows processing this point.
   * @return True if the point should be processed; false otherwise.
   */
  bool check_rate_limit();

  /**
   * @brief Store the latest camera data for Mode 0 (mono).
   * @param image_ptr Pointer to the image message.
   * @param camera_info_ptr Pointer to the camera info message.
   */
  void sync_callback_mode_0(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_ptr,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_ptr);

  /**
   * @brief Store the latest camera data for Mode 1 (stereo).
   * @param image_1_ptr Pointer to the first image message.
   * @param camera_info_1_ptr Pointer to the first camera info message.
   * @param image_2_ptr Pointer to the second image message.
   * @param camera_info_2_ptr Pointer to the second camera info message.
   */
  void sync_callback_mode_1(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_1_ptr,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_1_ptr,
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_2_ptr,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_2_ptr);

  /**
   * @brief Store the latest camera data for Mode 2 (mono+depth).
   * @param image_ptr Pointer to the image message.
   * @param camera_info_ptr Pointer to the camera info message.
   * @param depth_ptr Pointer to the depth message.
   */
  void sync_callback_mode_2(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_ptr,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_ptr,
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & depth_ptr);

  /**
   * @brief Call the SAM2 AddObjects service to segment an object at a point.
   *
   * @param initial_hint Initial hint for calling SAM2.
   * @return True if the service call was successful; false otherwise.
   */
  bool call_add_objects(const geometry_msgs::msg::Point initial_hint);

  /**
   * @brief Call the SAM2 RemoveObject service to remove an object from SAM2 memory.
   *
   * @return True if the service call was successful; false otherwise.
   */
  bool call_remove_object();

  /**
   * @brief Process the latest point and publish camera data.
   * @param point_msg The received point message.
   */
  void process_point(const geometry_msgs::msg::Point::SharedPtr point_msg);

  // Mode string
  std::string mode_;

  // True if using SAM2
  bool is_sam2_;

  // True if a point has been processed
  bool is_triggered_;

  // Depth format string
  std::string depth_format_string_;

  // Timeout for service call and discovery
  int service_call_timeout_;
  int service_discovery_timeout_;

  // Queue sizes
  int input_queue_size_;
  int output_queue_size_;
  int sync_queue_size_;

  // Rate limiting
  double max_rate_hz_;  // Maximum frequency in Hz
  std::chrono::time_point<std::chrono::steady_clock> last_point_time_;

  // Latest synchronized camera data for each mode
  struct
  {
    nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr image;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
    bool valid{false};
  } latest_mode_0_data_;

  struct
  {
    nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr image_1;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_1;
    nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr image_2;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_2;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
    bool valid{false};
  } latest_mode_1_data_;

  struct
  {
    nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr image;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info;
    nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr depth;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
    bool valid{false};
  } latest_mode_2_data_;
};

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia
