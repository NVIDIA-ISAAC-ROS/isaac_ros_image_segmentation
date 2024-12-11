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

#include <gmock/gmock.h>
#include "unet_decoder_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception


TEST(unet_decoder_node_test, test_empty_color_segmentation_mask_encoding)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("color_segmentation_mask_encoding", "");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::unet::UNetDecoderNode unet_decoder_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Received empty color segmentation mask encoding!"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(unet_decoder_node_test, test_invalid_color_segmentation_mask_encoding)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("color_segmentation_mask_encoding", "gbr8");
  options.append_parameter_override("color_palette", std::vector<int64_t>{1});
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::unet::UNetDecoderNode unet_decoder_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(
        e.what(),
        testing::HasSubstr("Received invalid color segmentation mask encoding"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(unet_decoder_node_test, test_empty_color_palette)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("color_segmentation_mask_encoding", "rgb8");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::unet::UNetDecoderNode unet_decoder_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(
        e.what(),
        testing::HasSubstr(
          "Received empty color palette! Fill this with a 24-bit hex color for each class!"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(unet_decoder_node_test, test_invalid_network_output_type)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  // options.arguments(
  // {
  //   "--ros-args",
  //   "-p", "color_segmentation_mask_encoding:='rgb8'",
  //   "-p", "color_palette:=[1]",
  //   "-p", "network_output_type:='invalid'",
  // });
  options.append_parameter_override("color_segmentation_mask_encoding", "rgb8");
  options.append_parameter_override("color_palette", std::vector<int64_t>(1));
  options.append_parameter_override("network_output_type", "invalid");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::unet::UNetDecoderNode unet_decoder_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Received invalid network output type: "));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
