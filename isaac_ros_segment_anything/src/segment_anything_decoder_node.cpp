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

#include "isaac_ros_segment_anything/segment_anything_decoder_node.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace segment_anything
{
namespace
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_COMPONENT_KEY[] = "segmentation_postprocessor/input_tensor";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw_rgb_f32";
constexpr char INPUT_TOPIC_NAME[] = "tensor_sub";

constexpr char RAW_OUTPUT_COMPONENT_KEY[] = "raw_segmentation_mask_sink/sink";
constexpr char RAW_OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw";
constexpr char RAW_OUTPUT_TOPIC_NAME[] = "segment_anything/raw_segmentation_mask";

constexpr char APP_YAML_FILENAME[] = "config/segment_anything_decoder_node.yaml";
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
    }}
};
#pragma GCC diagnostic pop
}  // namespace

SegmentAnythingDecoderNode::SegmentAnythingDecoderNode(const rclcpp::NodeOptions options)
: nitros::NitrosNode(
    options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  mask_width_(declare_parameter<int16_t>("mask_width", 960)),
  mask_height_(declare_parameter<int16_t>("mask_height", 544)),
  max_batch_size_(declare_parameter<int16_t>("max_batch_size", 20)),
  tensor_name_(declare_parameter<std::string>("tensor_name", ""))
{
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();
  startNitrosNode();
}

void SegmentAnythingDecoderNode::postLoadGraphCallback()
{
  const uint64_t block_size(mask_width_ * mask_height_ * max_batch_size_);

  getNitrosContext().setParameterUInt64(
    "segmentation_postprocessor",
    "nvidia::gxf::BlockMemoryPool", "block_size", block_size
  );
  getNitrosContext().setParameterStr(
    "segmentation_postprocessor",
    "nvidia::isaac_ros::SegmentAnythingPostprocessor",
    "tensor_name", tensor_name_
  );
}

SegmentAnythingDecoderNode::~SegmentAnythingDecoderNode() = default;

}  // namespace segment_anything
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode)
