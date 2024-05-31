// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_detection2_d_array_type/nitros_detection2_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{
namespace
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_COMPONENT_KEY[] = "sync/prompt_receiver";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_detection2_d_array";
constexpr char INPUT_TOPIC_NAME[] = "prompts";

constexpr char INPUT_IMAGE_COMPONENT_KEY[] = "sync/image_receiver";
constexpr char INPUT_IMAGE_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw_rgb_f32";
constexpr char INPUT_IMAGE_TOPIC_NAME[] = "tensor_pub";

constexpr char INPUT_MASK_COMPONENT_KEY[] = "sync/mask_receiver";
constexpr char INPUT_MASK_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw_rgb_f32";
constexpr char INPUT_MASK_TOPIC_NAME[] = "mask";

constexpr char RAW_OUTPUT_COMPONENT_KEY[] = "sink/output";
constexpr char RAW_OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw";
constexpr char RAW_OUTPUT_TOPIC_NAME[] = "tensor";

constexpr char APP_YAML_FILENAME[] = "config/segment_anything_data_encoder_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_segment_anything";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"gxf_isaac_ros_segment_anything", "gxf/lib/libgxf_isaac_ros_segment_anything.so"}};

const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {
  "config/segment_anything_spec_file.yaml"
};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_TOPIC_NAME,
    }},
  {RAW_OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = RAW_OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = RAW_OUTPUT_TOPIC_NAME,
      .frame_id_source_key = INPUT_COMPONENT_KEY,
    }},
  {INPUT_IMAGE_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_IMAGE_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_IMAGE_TOPIC_NAME,
    }},
  {INPUT_MASK_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_MASK_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_MASK_TOPIC_NAME,
    }}
};
#pragma GCC diagnostic pop
}  // namespace

bool IsSupportedInputPromptType(const std::string & network_output_type)
{
  return network_output_type == std::string{"bbox"} ||
         network_output_type == std::string{"point"};
}

SegmentAnythingDataEncoderNode::SegmentAnythingDataEncoderNode(const rclcpp::NodeOptions options)
: nitros::NitrosNode(
    options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
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

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection2DArray>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();
  startNitrosNode();
}

void SegmentAnythingDataEncoderNode::postLoadGraphCallback()
{
  getNitrosContext().setParameterStr(
    "prompt_processor",
    "nvidia::isaac_ros::SegmentAnythingPromptProcessor",
    "prompt_type_name", prompt_input_type_
  );

  getNitrosContext().setParameterInt32(
    "prompt_processor",
    "nvidia::isaac_ros::SegmentAnythingPromptProcessor",
    "max_batch_size", max_batch_size_
  );

  getNitrosContext().setParameterBool(
    "prompt_processor",
    "nvidia::isaac_ros::SegmentAnythingPromptProcessor",
    "has_input_mask", has_input_mask_
  );

  std::vector<int32_t> orig_img_dims(orig_img_dims_.begin(), orig_img_dims_.end());
  getNitrosContext().setParameter1DInt32Vector(
    "prompt_processor",
    "nvidia::isaac_ros::SegmentAnythingPromptProcessor",
    "orig_img_dim", orig_img_dims
  );
}

SegmentAnythingDataEncoderNode::~SegmentAnythingDataEncoderNode() = default;

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode)
