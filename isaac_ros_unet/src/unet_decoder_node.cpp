/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_unet/unet_decoder_node.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/sensor_msgs/image_encodings.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace unet
{
namespace
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_COMPONENT_KEY[] = "segmentation_postprocessor/input_tensor";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nhwc_rgb_f32";
constexpr char INPUT_TOPIC_NAME[] = "tensor_sub";

constexpr char RAW_OUTPUT_COMPONENT_KEY[] = "raw_segmentation_mask_vault/vault";
constexpr char RAW_OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_mono8";
constexpr char RAW_OUTPUT_TOPIC_NAME[] = "unet/raw_segmentation_mask";

constexpr char COLORED_OUTPUT_COMPONENT_KEY[] = "colored_segmentation_mask_vault/vault";
constexpr char COLORED_OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_rgb8";
constexpr char COLORED_OUTPUT_TOPIC_NAME[] = "unet/colored_segmentation_mask";

constexpr char APP_YAML_FILENAME[] = "config/unet_decoder_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_unet";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_nitros", "gxf/std/libgxf_std.so"},
  {"isaac_ros_nitros", "gxf/cuda/libgxf_cuda.so"},
  {"isaac_ros_unet", "gxf/segmentation_postprocessor/libgxf_segmentation_postprocessor.so"}};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_unet",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
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
  {COLORED_OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = COLORED_OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = COLORED_OUTPUT_TOPIC_NAME,
      .frame_id_source_key = INPUT_COMPONENT_KEY,
    }
  }
};
#pragma GCC diagnostic pop

bool IsSupportedNetworkOutputType(const std::string & network_output_type)
{
  return network_output_type == std::string{"softmax"} ||
         network_output_type == std::string{"sigmoid"} ||
         network_output_type == std::string{"argmax"};
}

const std::unordered_map<std::string, std::string> INPUT_FORMAT_TO_NITROS({
          {sensor_msgs::image_encodings::RGB8, nitros::nitros_image_rgb8_t::supported_type_name},
          {sensor_msgs::image_encodings::BGR8, nitros::nitros_image_bgr8_t::supported_type_name}
        });

}  // namespace

UNetDecoderNode::UNetDecoderNode(const rclcpp::NodeOptions options)
: nitros::NitrosNode(
    options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  color_segmentation_mask_encoding_(
    declare_parameter<std::string>("color_segmentation_mask_encoding", "rgb8")),
  color_palette_(
    declare_parameter<std::vector<int64_t>>("color_palette", std::vector<int64_t>({}))),
  network_output_type_(declare_parameter<std::string>("network_output_type", "softmax")),
  mask_width_(declare_parameter<int16_t>("mask_width", 960)),
  mask_height_(declare_parameter<int16_t>("mask_height", 544))
{
  // Received invalid color segmentation mask encoding
  if (color_segmentation_mask_encoding_.empty()) {
    RCLCPP_ERROR(
      get_logger(), "Received empty color segmentation mask encoding!");
    throw std::invalid_argument(
            "Received empty color segmentation mask encoding!");
  }


  auto nitros_format = INPUT_FORMAT_TO_NITROS.find(color_segmentation_mask_encoding_);
  if (nitros_format == std::end(INPUT_FORMAT_TO_NITROS)) {
    RCLCPP_ERROR(
      get_logger(), "Received invalid color segmentation mask encoding: %s",
      color_segmentation_mask_encoding_.c_str());
    throw std::invalid_argument(
            "Received invalid color segmentation mask encoding: " +
            color_segmentation_mask_encoding_);
  } else {
    config_map_[COLORED_OUTPUT_COMPONENT_KEY].compatible_data_format = nitros_format->second;
    config_map_[COLORED_OUTPUT_COMPONENT_KEY].use_compatible_format_only = true;
  }


  // Received empty color palette
  if (color_palette_.empty()) {
    RCLCPP_ERROR(
      get_logger(),
      "Received empty color palette! Fill this with a 24-bit hex color for each class!");
    throw std::invalid_argument(
            "Received empty color palette! Fill this with a 24-bit hex color for each class!");
  }

  // Received unsupported network output type
  if (!IsSupportedNetworkOutputType(network_output_type_)) {
    RCLCPP_ERROR(
      get_logger(),
      "Received invalid network output type: %s!",
      network_output_type_.c_str());
    throw std::invalid_argument("Received invalid network output type: " + network_output_type_);
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();

  startNitrosNode();
}

void UNetDecoderNode::postLoadGraphCallback()
{
  getNitrosContext().setParameter1DInt64Vector(
    "segmentation_mask_generator",
    "nvidia::isaac_ros::SegmentationMaskColorizer", "color_palette",
    color_palette_
  );

  getNitrosContext().setParameterStr(
    "segmentation_postprocessor",
    "nvidia::isaac_ros::SegmentationPostprocessor", "network_output_type",
    network_output_type_
  );

  const uint64_t mono_block_size{calculate_image_size(
      nitros::nitros_image_mono8_t::supported_type_name,
      mask_width_,
      mask_height_
    )};

  getNitrosContext().setParameterUInt64(
    "segmentation_postprocessor",
    "nvidia::gxf::BlockMemoryPool", "block_size", mono_block_size
  );

  std::string color_encoding{};
  if (color_segmentation_mask_encoding_ == sensor_msgs::image_encodings::RGB8) {
    color_encoding = nitros::nitros_image_rgb8_t::supported_type_name;
  } else if (color_segmentation_mask_encoding_ == sensor_msgs::image_encodings::BGR8) {
    color_encoding = nitros::nitros_image_bgr8_t::supported_type_name;
  } else {
    throw std::invalid_argument("Received unknown encoding!");
  }

  const uint64_t color_block_size{calculate_image_size(
      color_encoding, mask_width_, mask_height_
    )};

  getNitrosContext().setParameterUInt64(
    "segmentation_mask_generator",
    "nvidia::gxf::BlockMemoryPool", "block_size", color_block_size
  );
}

UNetDecoderNode::~UNetDecoderNode() = default;

}  // namespace unet
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::unet::UNetDecoderNode)
